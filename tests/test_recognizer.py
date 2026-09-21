# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Tests for the recognizer boundary: process facts in, enriched command text (or None) out.

The recognizer is the single testing seam for the model-identification feature: all rule
branching is tested here with plain data, without GPUs, psutil processes, or containers.
"""

import os
import sys
import time

import pytest

from nvitop.api import GpuProcess as GpuProcessBase
from nvitop.api import command_join, recognizer
from nvitop.api.process import is_modified_by_setproctitle
from nvitop.api.recognizer import Container, ProcessFacts, recognize
from nvitop.api.utils import NA
from nvitop.tui.library import GpuProcess


DOCKER_ID = 'a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890'


def _register_engine_rule(rule):
    recognizer.register_engine_rule(rule)


def _register_service_rule(rule):
    recognizer.register_service_rule(rule)


def _vllm_rule(facts):
    if facts.cmdline[:1] == ('VLLM::EngineCore0',):
        return 'vllm: Qwen2.5-72B-Instruct'
    return None


def _script_rule(facts):
    if len(facts.cmdline) >= 2 and facts.cmdline[1].endswith('.py'):
        return os.path.basename(facts.cmdline[1])
    return None


# ------------------------------------------------------------------------------
# Pure boundary: no rules registered -> always None (byte-identical fallback)
# ------------------------------------------------------------------------------


def test_recognize_without_rules_returns_none():
    assert recognize(ProcessFacts(cmdline=('python', 'main.py'))) is None
    assert recognize(ProcessFacts(cmdline=('VLLM::EngineCore0',))) is None


def test_facts_defaults():
    facts = ProcessFacts(cmdline=('python', 'main.py'))
    assert facts.ancestor_cmdlines == ()
    assert facts.cwd is None
    assert facts.container is None


# ------------------------------------------------------------------------------
# Pure boundary: label + attribution composition
# ------------------------------------------------------------------------------


def test_engine_label_gets_no_bare_directory():
    _register_engine_rule(_vllm_rule)
    facts = ProcessFacts(cmdline=('VLLM::EngineCore0',), cwd='/home/airroot/vllm')
    assert recognize(facts) == 'vllm: Qwen2.5-72B-Instruct'


def test_engine_label_with_named_container_and_cwd():
    _register_engine_rule(_vllm_rule)
    facts = ProcessFacts(
        cmdline=('VLLM::EngineCore0',),
        cwd='/workspace',
        container=Container(runtime='docker', identifier=DOCKER_ID, name='vllm-svc'),
    )
    assert recognize(facts) == 'vllm: Qwen2.5-72B-Instruct @ vllm-svc:/workspace'


def test_engine_label_with_unnamed_container_uses_short_id():
    _register_engine_rule(_vllm_rule)
    facts = ProcessFacts(
        cmdline=('VLLM::EngineCore0',),
        container=Container(runtime='docker', identifier=DOCKER_ID),
    )
    assert recognize(facts) == f'vllm: Qwen2.5-72B-Instruct @ {DOCKER_ID[:12]}'


def test_service_label_with_bare_directory():
    _register_service_rule(_script_rule)
    facts = ProcessFacts(cmdline=('python', 'main.py'), cwd='/home/airroot/octopus-api/')
    assert recognize(facts) == 'main.py @ octopus-api'


def test_service_label_with_container_and_cwd():
    _register_service_rule(_script_rule)
    facts = ProcessFacts(
        cmdline=('python', 'main.py'),
        cwd='/workspace',
        container=Container(runtime='lxd', identifier='tts-box'),
    )
    assert recognize(facts) == 'main.py @ tts-box:/workspace'


def test_service_label_with_container_without_cwd():
    _register_service_rule(_script_rule)
    facts = ProcessFacts(
        cmdline=('python', 'main.py'),
        container=Container(runtime='lxd', identifier='tts-box'),
    )
    assert recognize(facts) == 'main.py @ tts-box'


def test_engine_rule_wins_over_service_rule():
    _register_engine_rule(_vllm_rule)
    _register_service_rule(_script_rule)
    facts = ProcessFacts(cmdline=('VLLM::EngineCore0',), cwd='/srv')
    assert recognize(facts) == 'vllm: Qwen2.5-72B-Instruct'


def test_rule_returning_none_falls_through_to_next_rule():
    _register_service_rule(lambda facts: None)
    _register_service_rule(_script_rule)
    facts = ProcessFacts(cmdline=('python', 'main.py'), cwd='/srv/octopus')
    assert recognize(facts) == 'main.py @ octopus'


# ------------------------------------------------------------------------------
# Glue entry: switch, platform guard, exception normalization
# ------------------------------------------------------------------------------


class FakeProcess:
    pid = 4321

    def cmdline(self):
        return ['python', 'main.py']

    def parent(self):
        return None

    def cwd(self):
        return '/home/airroot/octopus-api'


class BrokenParentsProcess(FakeProcess):
    def parent(self):
        raise RuntimeError('psutil errors are normalized by the caller')

    def cwd(self):
        raise OSError('unreadable')


class GoneProcess(FakeProcess):
    def cmdline(self):
        raise RuntimeError('psutil errors are normalized by the caller')


def test_disabled_short_circuits_before_any_process_access():
    _register_service_rule(_script_rule)
    recognizer.set_enabled(False)
    assert recognizer.recognize_command(FakeProcess()) is None


def test_platform_guard_short_circuits():
    _register_service_rule(_script_rule)
    recognizer._PLATFORM = 'win32'
    assert recognizer.recognize_command(FakeProcess()) is None


def test_recognizes_through_glue_entry_on_linux(monkeypatch):
    _register_service_rule(_script_rule)
    recognizer._PLATFORM = 'linux'
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: None)
    assert recognizer.recognize_command(FakeProcess()) == 'main.py @ octopus-api'


def test_psutil_errors_in_parents_and_cwd_are_normalized(monkeypatch):
    _register_service_rule(_script_rule)
    recognizer._PLATFORM = 'linux'
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: None)
    # the cwd is unavailable, so the service label has no attribution and falls back
    assert recognizer.recognize_command(BrokenParentsProcess()) is None


def test_psutil_error_in_cmdline_yields_no_facts(monkeypatch):
    _register_service_rule(_script_rule)
    recognizer._PLATFORM = 'linux'
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: None)
    assert recognizer.recognize_command(GoneProcess()) is None


def test_facts_are_cached_per_pid(monkeypatch):
    _register_service_rule(_script_rule)
    recognizer._PLATFORM = 'linux'
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: None)
    calls = []
    original_cmdline = FakeProcess.cmdline

    def counting_cmdline(self):
        calls.append(self.pid)
        return original_cmdline(self)

    monkeypatch.setattr(FakeProcess, 'cmdline', counting_cmdline)
    assert recognizer.recognize_command(FakeProcess()) == 'main.py @ octopus-api'
    assert recognizer.recognize_command(FakeProcess()) == 'main.py @ octopus-api'
    assert calls == [FakeProcess.pid]


# ------------------------------------------------------------------------------
# Facts accessor (read side, used by the process detail views)
# ------------------------------------------------------------------------------


def test_process_facts_returns_gathered_facts(monkeypatch):
    recognizer._PLATFORM = 'linux'
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: None)
    facts = recognizer.process_facts(FakeProcess())
    assert facts is not None
    assert facts.cmdline == ('python', 'main.py')
    assert facts.cwd == '/home/airroot/octopus-api'


def test_process_facts_shares_the_cache_with_recognize_command(monkeypatch):
    _register_service_rule(_script_rule)
    recognizer._PLATFORM = 'linux'
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: None)
    assert recognizer.recognize_command(FakeProcess()) == 'main.py @ octopus-api'
    facts = recognizer.process_facts(FakeProcess())
    assert facts is not None
    assert facts.cwd == '/home/airroot/octopus-api'


def test_process_facts_is_none_when_disabled():
    recognizer.set_enabled(False)
    assert recognizer.process_facts(FakeProcess()) is None


def test_process_facts_is_none_on_unsupported_platform():
    recognizer._PLATFORM = 'win32'
    assert recognizer.process_facts(FakeProcess()) is None


def test_process_facts_is_none_for_a_gone_process(monkeypatch):
    recognizer._PLATFORM = 'linux'
    monkeypatch.setattr(recognizer, '_read_cgroup', lambda pid: None)
    assert recognizer.process_facts(GoneProcess()) is None


# ------------------------------------------------------------------------------
# The setproctitle predicate shared with `command_join`
# ------------------------------------------------------------------------------


def test_setproctitle_predicate_accepts_a_renamed_command_line():
    assert is_modified_by_setproctitle(['VLLM::EngineCore']) is True
    assert is_modified_by_setproctitle(('some title',)) is True


def test_setproctitle_predicate_rejects_real_command_lines():
    assert is_modified_by_setproctitle([]) is False
    assert is_modified_by_setproctitle(['python', 'main.py']) is False
    assert is_modified_by_setproctitle([sys.executable]) is False  # an existing file


# ------------------------------------------------------------------------------
# Caches: a failed gathering is retried, a successful one is kept
# ------------------------------------------------------------------------------


def test_negative_ttl_is_shorter_than_the_positive_ttl():
    cache = recognizer._TTLCache(ttl=30.0, negative_ttl=0.05)
    calls = []

    def failing_factory():
        calls.append('gather')
        return

    assert cache.get_or_put('pid', failing_factory) is None
    assert cache.get_or_put('pid', failing_factory) is None
    assert len(calls) == 1  # cached within the negative TTL

    time.sleep(0.06)
    assert cache.get_or_put('pid', lambda: 'facts') == 'facts'
    assert len(calls) == 1  # retried after the negative TTL, and this time it succeeded
    assert cache.get_or_put('pid', failing_factory) == 'facts'
    assert len(calls) == 1  # the successful value is kept for the long TTL


# ------------------------------------------------------------------------------
# Wiring: the TUI GpuProcess.command() honors the recognition; the API stays raw
# ------------------------------------------------------------------------------


class _FakeDevice:
    index = 0

    def memory_total(self):
        return NA

    def is_mig_device(self):
        return False


@pytest.fixture
def gpu_process():
    return GpuProcess(os.getpid(), device=_FakeDevice())


def test_gpu_process_command_falls_back_byte_identical(gpu_process):
    assert gpu_process.command() == command_join(gpu_process.cmdline())


def test_gpu_process_command_is_enriched_when_recognized(gpu_process):
    _register_engine_rule(_vllm_rule)
    recognizer._PLATFORM = 'linux'
    recognizer._FACTS_CACHE.put(gpu_process.pid, ProcessFacts(cmdline=('VLLM::EngineCore0',)))
    assert gpu_process.command() == 'vllm: Qwen2.5-72B-Instruct'


def test_host_snapshot_carries_enriched_command_and_original_cmdline(gpu_process):
    _register_engine_rule(_vllm_rule)
    recognizer._PLATFORM = 'linux'
    recognizer._FACTS_CACHE.put(gpu_process.pid, ProcessFacts(cmdline=('VLLM::EngineCore0',)))
    snapshot = gpu_process.host_snapshot()
    assert snapshot.command == 'vllm: Qwen2.5-72B-Instruct'
    assert snapshot.cmdline == gpu_process.cmdline()
    assert snapshot.command != command_join(snapshot.cmdline)


def test_batch_snapshots_carry_the_enriched_command(gpu_process):
    """The batch path feeds the panels and the one-shot output."""
    _register_engine_rule(_vllm_rule)
    recognizer._PLATFORM = 'linux'
    recognizer._FACTS_CACHE.put(gpu_process.pid, ProcessFacts(cmdline=('VLLM::EngineCore0',)))
    (snapshot,) = GpuProcess.take_snapshots([gpu_process], failsafe=True)
    assert snapshot.command == 'vllm: Qwen2.5-72B-Instruct'


def test_api_gpu_process_keeps_the_raw_command(gpu_process, monkeypatch):
    """The recognition is a TUI concern: the plain API keeps the original command text."""
    import nvitop.tui.library.process as tui_process_module

    calls = []

    def spy(process):
        calls.append(process)
        return 'enriched'

    monkeypatch.setattr(tui_process_module, 'recognize_command', spy)
    api_process = GpuProcessBase(os.getpid(), device=_FakeDevice())
    assert api_process.command() == command_join(api_process.cmdline())
    assert calls == []
