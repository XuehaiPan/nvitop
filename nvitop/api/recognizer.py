# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2026 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Recognition of the model identity and the service attribution for processes.

Inference engines often rename their worker processes with ``setproctitle`` (e.g. vLLM
renames them to ``VLLM::EngineCore``), so the command line no longer tells which model the
process is serving. Similarly, a bare ``python main.py`` tells nothing about which project
owns the process. This module recognizes the *model identity* and the *service
attribution* from process facts (the command line, the parent command lines, the working
directory, and the container the process lives in) and renders a richer command text for
such processes.

The public boundary is :func:`recognize`: process facts in, enriched text (or
:data:`None`) out. Rules are pluggable: engine rules and service rules are registered with
:func:`register_engine_rule` / :func:`register_service_rule`. Recognition never raises and
returns :data:`None` when nothing is recognized, so callers fall back to the original
command text. The whole feature can be turned off with :func:`set_enabled`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from nvitop.api.process import is_modified_by_setproctitle


__all__ = [
    'Container',
    'ProcessFacts',
    'enabled',
    'process_facts',
    'recognize',
    'recognize_command',
    'register_engine_rule',
    'register_service_rule',
    'set_enabled',
]


ProcessFactRule = Callable[['ProcessFacts'], Optional[str]]
"""A rule mapping process facts to a label (e.g. ``'vllm: Qwen2.5-72B-Instruct'``).

The rule returns :data:`None` when it does not apply. Engine rules match processes whose
identity is masked by renaming (e.g. ``setproctitle``); service rules match plain scripts
running under an interpreter.
"""


# ------------------------------------------------------------------------------
# Switch and platform guard
# ------------------------------------------------------------------------------

_PLATFORM: str = sys.platform
_ENABLED: bool = True


def set_enabled(value: bool) -> None:
    """Enable or disable the command recognition globally.

    When disabled, :func:`recognize_command` short-circuits and every caller keeps the
    original command text.
    """
    global _ENABLED
    _ENABLED = bool(value)


def enabled() -> bool:
    """Return whether the command recognition is enabled."""
    return _ENABLED


# ------------------------------------------------------------------------------
# Process facts
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Container:
    """The container a process lives in."""

    runtime: str  # 'docker' or 'lxd'
    identifier: str  # the container ID (Docker) or the payload name (LXD)
    name: str | None = None  # the resolved human-readable name, when available

    @property
    def display_name(self) -> str:
        """Return the name to show in the UI.

        LXD cgroup paths carry the payload name themselves; for Docker the resolved name
        is used when available, falling back to the short container ID.
        """
        if self.name:
            return self.name
        if self.runtime == 'lxd':
            return self.identifier
        return self.identifier[:12]


@dataclass(frozen=True)
class ProcessFacts:
    """Immutable facts about a process, gathered once and cached per PID.

    Everything except ``cmdline`` is best-effort: a field is :data:`None` when the
    information is unavailable (e.g. the process is gone, or reading it is denied).
    """

    cmdline: tuple[str, ...]
    ancestor_cmdlines: tuple[tuple[str, ...], ...] = ()
    cwd: str | None = None
    container: Container | None = None


# ------------------------------------------------------------------------------
# Rules and the recognition boundary
# ------------------------------------------------------------------------------

ENGINE_RULES: list[ProcessFactRule] = []
SERVICE_RULES: list[ProcessFactRule] = []


def register_engine_rule(rule: ProcessFactRule) -> ProcessFactRule:
    """Register an engine rule. Can be used as a decorator."""
    ENGINE_RULES.append(rule)
    return rule


def register_service_rule(rule: ProcessFactRule) -> ProcessFactRule:
    """Register a service rule. Can be used as a decorator."""
    SERVICE_RULES.append(rule)
    return rule


def _attribution(facts: ProcessFacts, *, include_bare_directory: bool) -> str | None:
    """Render the attribution part (after ``@``) of the enriched command text.

    A container attribution (``name:path``) applies to every recognized process; a bare
    directory attribution applies to service rules only, as an engine label already says
    what the process is.
    """
    if facts.container is not None:
        container = facts.container.display_name
        return f'{container}:{facts.cwd}' if facts.cwd else container
    if include_bare_directory and facts.cwd:
        directory = os.path.basename(os.path.normpath(facts.cwd))
        if directory:
            return directory
    return None


def recognize(facts: ProcessFacts) -> str | None:
    """Recognize the enriched command text from the given process facts.

    Engine rules are consulted first, then service rules. The first matching rule
    provides the label; the attribution part is composed from the container and the
    working directory. Returns :data:`None` when no rule matches.
    """
    label: str | None = None
    for rule in ENGINE_RULES:
        label = rule(facts)
        if label:
            break
    from_engine = label is not None
    if not from_engine:
        for rule in SERVICE_RULES:
            label = rule(facts)
            if label:
                break
    if not label:
        return None
    attribution = _attribution(facts, include_bare_directory=not from_engine)
    if attribution is None:
        if from_engine:
            return label  # the model identity alone is already informative
        return None  # a service label without any attribution is not worth showing
    return f'{label} @ {attribution}'


# ------------------------------------------------------------------------------
# Built-in rules
# ------------------------------------------------------------------------------


def _name_from_path(path_or_name: str) -> str:
    """Return the last path component of a model path or of a script path."""
    name = path_or_name.strip().replace('\\', '/').rstrip('/')
    return name.rsplit('/', 1)[-1]


# Model-name flags seen in the wild, both hyphen and underscore styles (paddleocr's
# genai_server uses ``--model_name`` / ``--model_dir``). Exact matches only, so
# ``--model`` never swallows ``--model_name``.
_MODEL_NAME_FLAGS = (
    '--served-model-name',
    '--served_model_name',
    '--model',
    '--model-id',
    '--model_id',
    '--model-name',
    '--model_name',
    '--model-dir',
    '--model_dir',
)


def _model_from_flags(argv: tuple[str, ...]) -> str | None:
    """Return the model name given by a ``--model``-style flag in *argv*.

    ``--served-model-name`` is preferred over the raw model path, as it is the name the
    deployment chose to present.
    """
    for flag in _MODEL_NAME_FLAGS:
        for index, arg in enumerate(argv):
            if arg == flag and index + 1 < len(argv):
                return argv[index + 1] or None
            if arg.startswith(flag + '='):
                return arg.split('=', 1)[1] or None
    return None


def _model_from_serve_positional(argv: tuple[str, ...]) -> str | None:
    """Return the model given as the positional argument of a ``... serve`` command."""
    if 'serve' not in argv:
        return None
    index = argv.index('serve') + 1
    while index < len(argv):
        arg = argv[index]
        if not arg.startswith('-'):
            return arg or None
        if '=' not in arg:  # a flag expecting a separate value: skip its value too
            index += 1
        index += 1
    return None


_OLLAMA_BLOB_RE = re.compile(r'blobs/(sha256-[0-9a-f]{64})')
_HF_CACHE_NAME_RE = re.compile(r'models--([^/\s]+?)--([^/\s]+)')
_MODEL_FILE_SUFFIXES = ('.gguf', '.safetensors')

# Common locations of the Ollama model store (its manifests map blobs to model names).
# ``$OLLAMA_MODELS`` of the nvitop process takes precedence when set.
_OLLAMA_MODEL_ROOTS: tuple[str, ...]
if os.environ.get('OLLAMA_MODELS'):
    _OLLAMA_MODEL_ROOTS = (os.environ['OLLAMA_MODELS'],)
else:
    _OLLAMA_MODEL_ROOTS = (
        '/usr/share/ollama/.ollama/models',
        '/var/lib/ollama/models',
        '/root/.ollama/models',
    )


def _ollama_blob_name(arg: str) -> str | None:
    """Resolve an Ollama blob path (``.../blobs/sha256-<digest>``) to its model name.

    The human-readable name lives in the manifests of the Ollama model store, whose
    location is probed from the common installation roots. Results are cached by the
    blob digest; an unreadable store yields :data:`None`.
    """
    match = _OLLAMA_BLOB_RE.search(arg)
    if match is None:
        return None
    digest = match.group(1)
    return _OLLAMA_NAME_CACHE.get_or_put(digest, lambda: _ollama_name_from_manifests(digest))


def _ollama_name_from_manifests(digest: str) -> str | None:
    """Search the manifests of the known Ollama model stores for a blob digest.

    A manifest is one small JSON file per model tag whose path ends with
    ``<registry>/<namespace>/<name>/<tag>``; the name shown is ``<name>:<tag>``.
    """
    for root in _OLLAMA_MODEL_ROOTS:
        manifests = os.path.join(root, 'manifests')
        for dirpath, _dirnames, filenames in os.walk(manifests):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                try:
                    with open(path, encoding='utf-8', errors='replace') as file:
                        content = file.read()
                except OSError:
                    continue  # unreadable store element: skip it
                if digest in content:
                    parts = os.path.relpath(path, manifests).split(os.sep)
                    if len(parts) >= 2:
                        return f'{parts[-2]}:{parts[-1]}'
    return None


def _model_path_hint(arg: str) -> str | None:
    """Return a model name hinted by a path-like command line argument."""
    blob = _ollama_blob_name(arg)
    if blob is not None:
        return blob
    match = _HF_CACHE_NAME_RE.search(arg)
    if match is not None:
        return f'{match.group(1)}/{match.group(2)}'
    if arg.lower().endswith(_MODEL_FILE_SUFFIXES):
        return _name_from_path(arg).rsplit('.', 1)[0] or None
    return None


def _model_from_cmdline(argv: tuple[str, ...]) -> str | None:
    """Extract a model name from a deployment command line.

    Recognized shapes, in order of trust: a ``--model``-style flag (``--model`` with a
    separate value or ``--model=...``, also ``--model-id`` / ``--model-name``, and the
    API-facing ``--served-model-name``), the positional argument of a ``... serve``
    command, and path-like arguments that carry a model identity (Ollama blobs,
    HuggingFace cache directories, model files). A flag value that points into an Ollama
    store is resolved through the manifests; when the store is unreadable, the process
    stays unrecognized rather than showing a raw digest.
    """
    for extractor in (_model_from_flags, _model_from_serve_positional):
        model = extractor(argv)
        if model is not None:
            resolved = _model_path_hint(model)
            if resolved is not None:
                return resolved
            if _OLLAMA_BLOB_RE.search(model):
                return None  # a raw digest is noise: leave the original text in place
            return _name_from_path(model)
    for arg in argv:
        model = _model_path_hint(arg)
        if model is not None:
            return model
    return None


def _model_from_ancestors(ancestor_cmdlines: tuple[tuple[str, ...], ...]) -> str | None:
    """Find a model name on the parent chain of a renamed engine process."""
    for argv in ancestor_cmdlines:
        if len(argv) <= 1:  # renamed or empty: no usable deployment command line
            continue
        model = _model_from_cmdline(argv)
        if model is not None:
            return model
    return None


@register_engine_rule
def engine_rule(facts: ProcessFacts) -> str | None:
    """Recognize engine processes renamed to the ``ENGINE::role`` convention.

    vLLM (``VLLM::EngineCore``), SGLang, ollama's vLLM workers and friends follow it.
    The engine label is the title prefix; the model is looked up on the parent chain,
    where the deployment command keeps its original command line.
    """
    if not is_modified_by_setproctitle(facts.cmdline):
        return None
    title = facts.cmdline[0]
    if '::' not in title:
        return None
    prefix = title.split('::', 1)[0].lower()
    if not prefix:
        return None
    model = _model_from_ancestors(facts.ancestor_cmdlines)
    if model is None:
        return None
    return f'{prefix}: {model}'


@register_service_rule
def python_script_rule(facts: ProcessFacts) -> str | None:
    """Recognize bare interpreter scripts (``python main.py`` / ``python -m <module>``).

    The label is the script (or module) name; the service attribution is composed from
    the working directory by the common composition, so the user can tell which project
    the process belongs to.
    """
    if len(facts.cmdline) < 2:
        return None
    interpreter = os.path.basename(facts.cmdline[0])
    if not interpreter.startswith('python'):
        return None
    entry = facts.cmdline[1]
    if entry == '-m':
        if len(facts.cmdline) < 3:
            return None
        return facts.cmdline[2].rsplit('.', 1)[-1] or None
    if entry.endswith('.py'):
        return _name_from_path(entry)
    return None


# ------------------------------------------------------------------------------
# Glue: TTL caches and facts gathering
# ------------------------------------------------------------------------------


class _TTLCache:
    """A minimal thread-safe TTL cache.

    Cached values may be :data:`None` (e.g. an unreadable process); a failed gathering is
    cached for *negative_ttl* only, so that it is retried within a few snapshot cycles
    instead of sticking for the whole TTL.
    """

    def __init__(self, ttl: float, negative_ttl: float) -> None:
        self._ttl = ttl
        self._negative_ttl = negative_ttl
        self._entries: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get_or_put(self, key: Any, factory: Callable[[], Any]) -> Any:
        """Return the cached value for *key*, gathering it with *factory* when stale."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                ttl = self._negative_ttl if entry[1] is None else self._ttl
                if now - entry[0] < ttl:
                    return entry[1]
        value = factory()  # computed outside the lock, so a slow factory never blocks
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
        return value

    def put(self, key: Any, value: Any) -> None:
        """Seed the cache with a known value."""
        with self._lock:
            self._entries[key] = (time.monotonic(), value)

    def clear(self) -> None:
        """Drop every cached entry."""
        with self._lock:
            self._entries.clear()


# The facts of a process (its command line, its parent chain, its working directory and
# its container) change on the scale of deployments, not of snapshots. A few snapshot
# cycles keep them fresh enough, while an unreadable process is retried quickly.
_FACTS_CACHE = _TTLCache(ttl=10.0, negative_ttl=2.0)
_CONTAINER_NAME_CACHE = _TTLCache(ttl=3600.0, negative_ttl=60.0)  # keyed by container ID
_OLLAMA_NAME_CACHE = _TTLCache(ttl=3600.0, negative_ttl=60.0)  # keyed by blob digest

_MAX_ANCESTOR_HOPS = 4


def _safe_call(func: Callable[[], Any]) -> Any:
    """Call a best-effort facts getter, normalizing every failure to :data:`None`."""
    try:
        return func()
    except Exception:  # noqa: BLE001 - a failed field must never fail the whole facts
        return None


def _ancestor_cmdlines(
    process: Any,
    max_hops: int = _MAX_ANCESTOR_HOPS,
) -> tuple[tuple[str, ...], ...]:
    """Walk up the parent chain and collect the command lines (best-effort)."""
    cmdlines = []
    current: Any = process
    for _ in range(max_hops):
        current = _safe_call(current.parent)
        if current is None:
            break
        cmdline = _safe_call(current.cmdline)
        if not cmdline:
            break
        cmdlines.append(tuple(cmdline))
    return tuple(cmdlines)


def _read_cgroup(pid: int) -> str | None:
    """Read ``/proc/<pid>/cgroup``. Returns :data:`None` when unavailable."""
    try:
        with open(f'/proc/{pid}/cgroup', encoding='utf-8', errors='replace') as file:
            return file.read()
    except (OSError, ValueError):
        return None


_DOCKER_ID_RE = r'([0-9a-f]{64})'
_DOCKER_CGROUP_PATTERNS = (
    re.compile(rf'/docker/{_DOCKER_ID_RE}'),  # cgroupfs driver
    re.compile(rf'/docker-{_DOCKER_ID_RE}\.scope'),  # systemd driver
)
_LXD_CGROUP_PATTERN = re.compile(r'/lxc\.payload\.([^\s/:]+)')


def _detect_container(cgroup: str | None) -> Container | None:
    """Map ``/proc/<pid>/cgroup`` content to a :class:`Container`.

    Docker (both the cgroupfs and the systemd driver layouts) and LXD containers are
    detected; Kubernetes pods and unknown runtimes return :data:`None`.
    """
    if not cgroup or 'kubepods' in cgroup:
        return None  # outside a supported container (Kubernetes pods are out of scope)
    for pattern in _DOCKER_CGROUP_PATTERNS:
        match = pattern.search(cgroup)
        if match is not None:
            return Container(runtime='docker', identifier=match.group(1))
    match = _LXD_CGROUP_PATTERN.search(cgroup)
    if match is not None:
        return Container(runtime='lxd', identifier=match.group(1))
    return None


def _docker_container_name(container_id: str) -> str | None:
    """Look up the name of a Docker container with the Docker CLI (best-effort)."""
    executable = shutil.which('docker')
    if executable is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            (executable, 'inspect', '--format', '{{.Name}}', container_id),
            check=False,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = result.stdout.strip().lstrip('/')
    return name or None


def _container_of(pid: int) -> Container | None:
    """Detect the container a process lives in and resolve its name (best-effort)."""
    container = _detect_container(_read_cgroup(pid))
    if container is None or container.runtime == 'lxd':
        return container  # LXD cgroup paths carry the container name themselves
    name = _CONTAINER_NAME_CACHE.get_or_put(
        container.identifier,
        lambda: _docker_container_name(container.identifier),
    )
    if name is None:
        return container
    return replace(container, name=name)


def _gather_facts(process: Any) -> ProcessFacts | None:
    """Collect :class:`ProcessFacts` for a process (best-effort, never raises)."""
    try:
        cmdline = tuple(process.cmdline())
        if not cmdline:
            return None
        return ProcessFacts(
            cmdline=cmdline,
            ancestor_cmdlines=_ancestor_cmdlines(process),
            cwd=_safe_call(process.cwd),
            container=_container_of(process.pid),
        )
    except Exception:  # noqa: BLE001 - recognition must never break the display
        return None


def process_facts(process: Any) -> ProcessFacts | None:
    """Return the cached facts for a process, gathering them on the first access.

    Returns :data:`None` when the recognition is disabled or the platform is unsupported.
    This is the read-side companion of :func:`recognize_command`, e.g. for detail views
    that want the container of the process without re-gathering it.
    """
    if not _ENABLED or not _PLATFORM.startswith('linux'):
        return None
    return _FACTS_CACHE.get_or_put(process.pid, lambda: _gather_facts(process))


def recognize_command(process: Any) -> str | None:
    """Recognize the enriched command text for a process (never raises).

    Returns :data:`None` — meaning *not recognized* — when the recognition is disabled,
    the platform is unsupported, or no rule matches. Callers fall back to the original
    command text, so a :data:`None` result is always safe.
    """
    facts = process_facts(process)
    if facts is None:
        return None
    try:
        return recognize(facts)
    except Exception:  # noqa: BLE001 - recognition must never break the display
        return None
