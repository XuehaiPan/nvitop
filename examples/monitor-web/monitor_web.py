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
"""Minimal stdlib HTTP(S) GPU dashboard built on ``nvitop``.

Drives :func:`nvitop.collect_in_background` on a daemon thread, stores the samples in a rotating
ring buffer (24h by default), and serves a small browser dashboard plus JSON endpoints
(``/metrics.json``, ``/history.json``) over either HTTP or HTTPS using only the Python standard
library.
"""

from __future__ import annotations

import argparse
import http.server
import json
import math
import os
import re
import signal
import socket
import ssl
import sys
import threading
import time
import urllib.parse
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TextIO, TypedDict

from nvitop import (
    Device,
    GiB,
    MiB,
    ResourceMetricCollector,
    bytes2human,
    collect_in_background,
    colored,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


__all__ = ['main']


DEFAULT_RETENTION_SECONDS = 24 * 3600
_MIN_INTERVAL = 0.25

_DURATION_RE = re.compile(
    r'\A\s*(?P<value>\d+(?:\.\d+)?)\s*'
    r'(?P<unit>s|sec|secs|second|seconds|'
    r'm|min|mins|minute|minutes|'
    r'h|hr|hrs|hour|hours|'
    r'd|day|days)?\s*\Z',
    re.IGNORECASE,
)
_DURATION_MULTIPLIERS = {
    's': 1.0,
    'sec': 1.0,
    'secs': 1.0,
    'second': 1.0,
    'seconds': 1.0,
    'm': 60.0,
    'min': 60.0,
    'mins': 60.0,
    'minute': 60.0,
    'minutes': 60.0,
    'h': 3600.0,
    'hr': 3600.0,
    'hrs': 3600.0,
    'hour': 3600.0,
    'hours': 3600.0,
    'd': 86400.0,
    'day': 86400.0,
    'days': 86400.0,
}


class Sample(NamedTuple):
    """A collector sample: an epoch timestamp and the metric mapping captured at that instant."""

    epoch: float
    metrics: dict[str, float]


class BufferStats(TypedDict):
    """Shape of :meth:`MetricStore.stats` — also the ``buffer`` field on the JSON payloads."""

    count: int
    max_count: int
    retention_seconds: float
    retention_human: str
    oldest_epoch: float
    newest_epoch: float


# Reference: https://stackoverflow.com/a/28950776
def get_ip_address() -> str:
    """Best-effort guess of a routable local IPv4 address; silently falls back to ``127.0.0.1``."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.0)
    try:
        # Doesn't even have to be reachable; the kernel picks the source IP for the route.
        s.connect(('10.254.254.254', 1))
        return str(s.getsockname()[0])
    except OSError:
        # Non-blocking UDP probe surfaces every failure (no route, gaierror, EHOSTUNREACH, ...)
        # as an OSError subclass; collapse them all into the loopback fallback.
        return '127.0.0.1'
    finally:
        s.close()


def parse_duration(text: str) -> float:
    """Parse a human-friendly duration into seconds.

    Accepts ``s``, ``m``/``min``, ``h``/``hour``, and ``d``/``day`` suffixes (case-insensitive).
    A bare number is treated as seconds, so ``'600'`` is equivalent to ``'600s'``.
    """
    match = _DURATION_RE.match(text)
    if match is None:
        raise argparse.ArgumentTypeError(f'Invalid duration: {text!r}')
    unit = (match.group('unit') or 's').lower()
    seconds = float(match.group('value')) * _DURATION_MULTIPLIERS[unit]
    if seconds <= 0:
        raise argparse.ArgumentTypeError(f'Invalid duration: {text!r}')
    return seconds


def format_duration(seconds: float) -> str:
    """Render ``seconds`` using the largest exact unit (``1d``, ``12h``, ``30m``, ``45s``)."""
    for unit, multiplier in (('d', 86400.0), ('h', 3600.0), ('m', 60.0)):
        if seconds >= multiplier and seconds % multiplier == 0:
            return f'{int(seconds / multiplier)}{unit}'
    return f'{seconds:g}s'


def cprint(text: str = '', *, file: TextIO | None = None) -> None:
    """Print a line, applying a bold color to any leading ``INFO:`` / ``WARNING:`` / ``ERROR:`` prefix."""
    for prefix, color in (('INFO: ', 'green'), ('WARNING: ', 'yellow'), ('ERROR: ', 'red')):
        if text.startswith(prefix):
            text = text.replace(
                prefix.rstrip(),
                colored(prefix.rstrip(), color=color, attrs=('bold',)),  # type: ignore[arg-type]
                1,
            )
            break
    print(text, file=file)


class MetricStore:
    """Thread-safe rotating buffer of collector samples.

    Each entry is ``(timestamp, metrics_dict)``. The deque, the closed flag, and the last collector
    error are guarded by an internal lock. :meth:`history` snapshots the deque under the lock and
    processes the copy outside the lock; per-sample metric mappings are defensively copied on
    insertion, so callers cannot mutate stored entries.

    The buffer keeps at most ``max(1, int(retention / interval))`` samples; older entries are
    evicted automatically by :class:`deque`.
    """

    def __init__(self, *, retention_seconds: float, interval: float) -> None:
        """Build an empty buffer sized for ``retention_seconds`` at ``interval`` per sample."""
        maxlen = max(1, int(retention_seconds / interval))
        self._lock = threading.Lock()
        self._retention_seconds = retention_seconds
        self._samples: deque[Sample] = deque(maxlen=maxlen)
        self._closed = False
        self._last_error: str | None = None

    def update(self, metrics: Mapping[str, float]) -> None:
        """Append one sample; oldest entries are evicted by the deque ``maxlen``."""
        sample = Sample(epoch=time.time(), metrics=dict(metrics))
        with self._lock:
            self._samples.append(sample)
            self._last_error = None

    def latest(self) -> Sample | None:
        """Return the most recent sample, or :data:`None` if the buffer is empty."""
        with self._lock:
            return self._samples[-1] if self._samples else None

    def history(
        self,
        *,
        bucket_seconds: float | None = None,
        limit: int | None = None,
        max_samples: int | None = None,
        since: float | None = None,
    ) -> list[Sample]:
        """Snapshot copy of the buffer, optionally filtered, trimmed, and downsampled."""
        with self._lock:
            samples = list(self._samples)
        if since is not None:
            samples = [s for s in samples if s.epoch > since]
        if limit is not None and len(samples) > limit:
            samples = samples[-limit:]
        return _downsample_history(samples, max_samples, bucket_seconds=bucket_seconds)

    def stats(self) -> BufferStats:
        """Return buffer statistics suitable for embedding in the JSON payload."""
        with self._lock:
            count = len(self._samples)
            oldest = self._samples[0].epoch if count else 0.0
            newest = self._samples[-1].epoch if count else 0.0
            max_count = self._samples.maxlen or 0
        return BufferStats(
            count=count,
            max_count=max_count,
            retention_seconds=self._retention_seconds,
            retention_human=format_duration(self._retention_seconds),
            oldest_epoch=oldest,
            newest_epoch=newest,
        )

    def record_error(self, message: str) -> None:
        """Record the most recent collector failure for surfacing through ``/metrics.json``."""
        with self._lock:
            self._last_error = message

    def last_error(self) -> str | None:
        """Return the most recent collector failure message, cleared by the next successful sample."""
        with self._lock:
            return self._last_error

    def close(self) -> None:
        """Mark the store closed so the next ``on_collect`` callback returns :data:`False`."""
        with self._lock:
            self._closed = True

    def is_closed(self) -> bool:
        """Return :data:`True` once :meth:`close` has been called."""
        with self._lock:
            return self._closed


HTML_PATH = Path(__file__).resolve().with_suffix('.html')


class MonitorServer(http.server.ThreadingHTTPServer):
    """:class:`ThreadingHTTPServer` subclass that owns the dashboard configuration.

    Configuration (the metric store, device descriptions, displayed hostname, sampling interval)
    lives on the server instance so each request handler can read it via ``self.server`` rather
    than relying on globally mutated class state. This also makes it trivial to run more than one
    dashboard in the same process — for tests, for example.
    """

    allow_reuse_address = True

    def __init__(  # pylint: disable=too-many-arguments
        self,
        server_address: tuple[str, int],
        *,
        store: MetricStore,
        devices_info: list[dict[str, Any]],
        hostname: str,
        interval: float,
    ) -> None:
        """Bind to ``server_address`` and remember the dashboard configuration."""
        self.store = store
        self.devices_info = devices_info
        self.hostname = hostname
        self.interval = interval
        super().__init__(server_address, MonitorRequestHandler)


class MonitorRequestHandler(http.server.BaseHTTPRequestHandler):
    """Tiny request router serving the dashboard HTML and JSON snapshots."""

    server_version = 'nvitop-monitor-web'
    sys_version = ''

    if TYPE_CHECKING:
        # Narrow the inherited `server` attribute so route handlers get IDE/type-checker support
        # when reading the dashboard configuration via `self.server.<field>`.
        server: MonitorServer

    def log_message(self, *_args: Any, **_kwargs: Any) -> None:
        """Silence the default per-request access log."""

    # pylint: disable-next=invalid-name
    def do_GET(self) -> None:
        """Dispatch GET routes."""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/':
            self._send_html()
        elif parsed.path == '/metrics.json':
            self._send_metrics_json()
        elif parsed.path == '/history.json':
            self._send_history_json(parsed.query)
        else:
            self._send_404()

    def _safe_write(self, body: bytes) -> None:
        """Write ``body`` to the response, swallowing client-disconnect errors."""
        # Routine browser refreshes drop the connection mid-write; suppress these so the per-thread
        # error handler in the standard library does not print a traceback for each one.
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_html(self) -> None:
        """Serve the dashboard HTML; respond ``500`` if the on-disk asset is missing or unreadable."""
        try:
            body = HTML_PATH.read_bytes()
        except OSError:
            self._send_500(b'500 HTML asset unavailable\n')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def _send_metrics_json(self) -> None:
        """Serve the latest collector sample plus dashboard metadata as strict JSON."""
        store = self.server.store
        interval = self.server.interval
        latest = store.latest()
        sample_time = latest.epoch if latest is not None else 0.0
        metrics = latest.metrics if latest is not None else {}
        now = time.time()
        stale_seconds = max(0.0, now - sample_time) if latest is not None else None
        collector_error = store.last_error()
        if collector_error is not None:
            status = 'failed'
        elif latest is None:
            status = 'warming_up'
        elif stale_seconds is not None and stale_seconds > 2 * interval:
            status = 'stalled'
        else:
            status = 'ready'
        payload = {
            'interval': interval,
            'hostname': self.server.hostname,
            'server_time': now,
            'sample_time': sample_time,
            'stale_seconds': stale_seconds,
            'status': status,
            'collector_error': collector_error,
            'buffer': store.stats(),
            'devices': self.server.devices_info,
            'metrics': metrics,
            'metrics_human': _humanize_metrics(metrics),
        }
        self._send_json(payload)

    def _send_history_json(self, query: str) -> None:
        """Serve filtered/downsampled history as strict JSON; respond ``400`` on bad query input."""
        params = urllib.parse.parse_qs(query)
        try:
            bucket_seconds = _parse_positive_float(params, 'bucket_seconds')
            limit = _parse_positive_int(params, 'limit')
            max_samples = _parse_positive_int(params, 'max_samples')
            since = _parse_finite_float(params, 'since')
        except _BadRequestError as ex:
            self._send_400(f'400 Bad Request: {ex}\n'.encode())
            return
        store = self.server.store
        history = store.history(
            bucket_seconds=bucket_seconds,
            limit=limit,
            max_samples=max_samples,
            since=since,
        )
        payload = {
            'buffer': store.stats(),
            'samples': [{'epoch': sample.epoch, 'metrics': sample.metrics} for sample in history],
        }
        self._send_json(payload)

    def _send_json(self, payload: object) -> None:
        """Encode ``payload`` as strict JSON (200 OK) after coercing non-finite floats to null."""
        # Strict JSON has no representation for NaN/Infinity, so `allow_nan=False` would raise.
        # `_finite()` first maps non-finite floats (nan/+inf/-inf) to None; the collector emits
        # NaN for any metric key seen previously but absent from the current snapshot.
        body = json.dumps(_finite(payload), allow_nan=False, default=float).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def _send_400(self, body: bytes) -> None:
        """Respond ``400 Bad Request`` with a plain-text body."""
        self.send_response(400)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def _send_404(self) -> None:
        """Respond ``404 Not Found`` with a short plain-text body."""
        body = b'404 Not Found\n'
        self.send_response(404)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self._safe_write(body)

    def _send_500(self, body: bytes) -> None:
        """Respond ``500 Internal Server Error`` with the provided plain-text body."""
        self.send_response(500)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self._safe_write(body)


def _finite(value: Any) -> Any:
    """Recursively replace non-finite floats (``nan``/``+inf``/``-inf``) with :data:`None`.

    Strict JSON has no representation for ``NaN`` or ``Infinity``, so the encoder is invoked with
    ``allow_nan=False``. The ``nvitop`` collector writes :data:`math.nan` for any metric key that
    was sampled previously but is absent from the current snapshot (see ``_MetricBuffer.add`` in
    ``nvitop/api/collector.py``); this function maps those values to :data:`None` so the encoder
    accepts them.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _downsample_history(
    samples: list[Sample],
    max_samples: int | None,
    *,
    bucket_seconds: float | None = None,
) -> list[Sample]:
    """Downsample ``samples`` to at most ``max_samples`` entries.

    Returns ``samples`` unchanged when ``max_samples`` is :data:`None` or the input already fits.
    When ``bucket_seconds`` is given, groups samples into epoch-aligned time buckets and averages
    each bucket, doubling ``bucket_seconds`` until the result fits ``max_samples``. Otherwise the
    input is split into ``max_samples`` equal-count slices and each slice is averaged.
    """
    if not samples:
        return []
    if max_samples is None or len(samples) <= max_samples:
        return samples
    if bucket_seconds is not None:
        downsampled = _average_history_time_buckets(samples, bucket_seconds=bucket_seconds)
        while len(downsampled) > max_samples:
            factor = math.ceil(len(downsampled) / max_samples)
            bucket_seconds *= factor
            downsampled = _average_history_time_buckets(samples, bucket_seconds=bucket_seconds)
        return downsampled

    if max_samples <= 1:
        return [_average_history_bucket(samples)]

    sample_count = len(samples)
    downsampled = []
    for bucket_index in range(max_samples):
        start = bucket_index * sample_count // max_samples
        stop = (bucket_index + 1) * sample_count // max_samples
        downsampled.append(_average_history_bucket(samples[start:stop]))
    return downsampled


def _average_history_time_buckets(samples: list[Sample], *, bucket_seconds: float) -> list[Sample]:
    buckets: dict[float, list[Sample]] = {}
    for sample in samples:
        bucket_start = math.floor(sample.epoch / bucket_seconds) * bucket_seconds
        buckets.setdefault(bucket_start, []).append(sample)
    return [
        _average_history_bucket(bucket, timestamp=bucket_start)
        for bucket_start, bucket in sorted(buckets.items())
    ]


def _average_history_bucket(samples: list[Sample], *, timestamp: float | None = None) -> Sample:
    if timestamp is None:
        timestamp = sum(sample.epoch for sample in samples) / len(samples)
    keys = {key for sample in samples for key in sample.metrics}
    metrics_sum = dict.fromkeys(keys, 0.0)
    metrics_count = dict.fromkeys(keys, 0)
    for sample in samples:
        for key, value in sample.metrics.items():
            if isinstance(value, (float, int)) and math.isfinite(value):
                metrics_sum[key] += float(value)
                metrics_count[key] += 1

    # Skip keys whose entire bucket was non-finite; the JS treats absent keys as gaps the same way
    # it treats null, so dropping them saves bytes on 24-h downsampled payloads.
    metrics_average = {
        key: metrics_sum[key] / metrics_count[key] for key in keys if metrics_count[key] > 0
    }
    return Sample(epoch=timestamp, metrics=metrics_average)


def _humanize_metrics(metrics: dict[str, float]) -> dict[str, str]:
    human: dict[str, str] = {}
    for key, value in metrics.items():
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            continue
        if ' (MiB)' in key:
            human[key] = bytes2human(value * MiB, min_unit=MiB)
        elif ' (GiB)' in key:
            human[key] = bytes2human(value * GiB, min_unit=GiB)
    return human


class _BadRequestError(ValueError):
    """Raised when a query-parameter value is present but unparsable or out of range."""


def _query_value(params: dict[str, list[str]], name: str) -> str | None:
    """Return the first value for ``name`` in ``params`` (or :data:`None` if absent)."""
    values = params.get(name)
    return values[0] if values else None


def _parse_positive_int(params: dict[str, list[str]], name: str) -> int | None:
    """Return a strictly positive ``int``; raise :class:`_BadRequestError` if invalid."""
    text = _query_value(params, name)
    if text is None:
        return None
    try:
        value = int(text)
    except ValueError as ex:
        raise _BadRequestError(f'`{name}` expected positive integer, got {text!r}') from ex
    if value <= 0:
        raise _BadRequestError(f'`{name}` expected positive integer, got {value}')
    return value


def _parse_finite_float(params: dict[str, list[str]], name: str) -> float | None:
    """Return a finite ``float``; raise :class:`_BadRequestError` if invalid."""
    text = _query_value(params, name)
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError as ex:
        raise _BadRequestError(f'`{name}` expected finite float, got {text!r}') from ex
    if not math.isfinite(value):
        raise _BadRequestError(f'`{name}` expected finite float, got {text!r}')
    return value


def _parse_positive_float(params: dict[str, list[str]], name: str) -> float | None:
    """Return a strictly positive finite ``float``; raise :class:`_BadRequestError` if invalid."""
    value = _parse_finite_float(params, name)
    if value is not None and value <= 0:
        raise _BadRequestError(f'`{name}` expected positive float, got {value}')
    return value


def build_ssl_context(args: argparse.Namespace) -> ssl.SSLContext | None:
    """Build an :class:`ssl.SSLContext` from the parsed args, or :data:`None` for plain HTTP.

    Raises :class:`SystemExit` with a friendly ``ERROR:`` line if the certificate, private key,
    or trusted-CA bundle cannot be loaded (malformed PEM, key/cert mismatch, passphrase-protected
    key, etc.) — :func:`parse_arguments` only verifies file existence, not parse-ability.
    """
    if args.certfile is None and args.keyfile is None:
        return None
    # `parse_arguments()` enforces that `--certfile`/`--keyfile` are paired and that the mTLS flags
    # (`--client-cafile`/`--client-capath` + `--client-auth-required`) come as a set.
    assert args.certfile is not None
    assert args.keyfile is not None
    ctx = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
    try:
        ctx.load_cert_chain(certfile=args.certfile, keyfile=args.keyfile)
    except (ssl.SSLError, OSError) as ex:
        raise SystemExit(
            f'ERROR: Failed to load TLS certificate/key from '
            f'`{args.certfile}` / `{args.keyfile}`: {ex}',
        ) from ex
    if args.client_auth_required:
        try:
            ctx.load_verify_locations(cafile=args.client_cafile, capath=args.client_capath)
        except (ssl.SSLError, OSError) as ex:
            raise SystemExit(
                f'ERROR: Failed to load client CA bundle from '
                f'`{args.client_cafile or args.client_capath}`: {ex}',
            ) from ex
        ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the web monitor."""

    def posfloat(arg: str) -> float:
        value = float(arg)
        if not math.isfinite(value) or value <= 0:
            raise ValueError
        return value

    posfloat.__name__ = 'positive float'

    parser = argparse.ArgumentParser(
        description='Minimal stdlib HTTP(S) GPU dashboard built on `nvitop`.',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        '--help',
        '-h',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.',
    )
    parser.add_argument(
        '--hostname',
        '--host',
        '-H',
        dest='hostname',
        type=str,
        default=get_ip_address(),
        metavar='HOSTNAME',
        help='Hostname to display in the dashboard. (default: %(default)s)',
    )
    parser.add_argument(
        '--bind-address',
        '--bind',
        '-B',
        dest='bind_address',
        type=str,
        default='127.0.0.1',
        metavar='ADDRESS',
        help='Local address to bind to. (default: %(default)s)',
    )
    parser.add_argument(
        '--port',
        '-p',
        type=int,
        default=5555,
        help='Port to listen on. (default: %(default)d)',
    )
    parser.add_argument(
        '--interval',
        type=posfloat,
        default=1.0,
        metavar='SEC',
        help='Interval between collector samples in seconds. (default: %(default)s)',
    )
    parser.add_argument(
        '--retention',
        type=parse_duration,
        default=DEFAULT_RETENTION_SECONDS,
        metavar='DURATION',
        help=(
            'Buffer retention duration. Accepts `s`/`m`/`h`/`d` suffixes\n'
            '(e.g. `90s`, `30min`, `12h`, `1d`). Default: 1d.'
        ),
    )

    tls = parser.add_argument_group('TLS / mTLS options')
    tls.add_argument(
        '--certfile',
        type=str,
        default=None,
        metavar='PATH',
        help=(
            'Path to the TLS certificate file (PEM).\n'
            'Enables HTTPS when set together with `--keyfile`.'
        ),
    )
    tls.add_argument(
        '--keyfile',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to the TLS private key file (PEM).\nRequired if `--certfile` is set.',
    )
    tls.add_argument(
        '--client-cafile',
        dest='client_cafile',
        type=str,
        default=None,
        metavar='PATH',
        help=(
            'Path to a PEM bundle of trusted client CA certificates for mutual TLS.\n'
            'Must be passed together with `--client-auth-required`.'
        ),
    )
    tls.add_argument(
        '--client-capath',
        dest='client_capath',
        type=str,
        default=None,
        metavar='PATH',
        help=(
            'Path to a directory of trusted client CA certificates for mutual TLS.\n'
            'Must be passed together with `--client-auth-required`.'
        ),
    )
    tls.add_argument(
        '--client-auth-required',
        dest='client_auth_required',
        action='store_true',
        help=(
            'Require clients to present a valid certificate (mutual TLS).\n'
            'Must be passed together with `--client-cafile` or `--client-capath`.'
        ),
    )

    args = parser.parse_args()

    if args.interval < _MIN_INTERVAL:
        parser.error(
            f'`--interval` value {args.interval:0.2g}s is too short, '
            f'which may cause performance issues. Expected `{_MIN_INTERVAL}` or higher.',
        )

    if (args.certfile is None) != (args.keyfile is None):
        parser.error('`--certfile` and `--keyfile` must be specified together.')
    if args.certfile is not None and not os.path.isfile(args.certfile):
        parser.error(f'`--certfile` not found: {args.certfile}')
    if args.keyfile is not None and not os.path.isfile(args.keyfile):
        parser.error(f'`--keyfile` not found: {args.keyfile}')
    if args.client_cafile is not None and not os.path.isfile(args.client_cafile):
        parser.error(f'`--client-cafile` not found: {args.client_cafile}')
    if args.client_capath is not None and not os.path.isdir(args.client_capath):
        parser.error(f'`--client-capath` not a directory: {args.client_capath}')

    ca_provided = args.client_cafile is not None or args.client_capath is not None
    if (ca_provided or args.client_auth_required) and args.certfile is None:
        parser.error('Mutual TLS options require `--certfile` and `--keyfile`.')
    if ca_provided != args.client_auth_required:
        parser.error(
            '`--client-cafile` / `--client-capath` and `--client-auth-required` must be '
            'specified together to enable mutual TLS.',
        )

    return args


def _describe_devices(devices: Sequence[Device]) -> list[dict[str, Any]]:
    info: list[dict[str, Any]] = []
    for device in devices:
        memory_total = device.memory_total()
        memory_total_mib = (
            int(memory_total) // (1024 * 1024) if isinstance(memory_total, int) else 0
        )
        uuid = device.uuid()
        info.append(
            {
                'index': device.physical_index,
                'name': str(device.name()),
                'memory_total_mib': memory_total_mib,
                'memory_total_human': bytes2human(memory_total),
                'uuid': uuid if isinstance(uuid, str) else None,
            },
        )
    return info


def main() -> int:  # pylint: disable=too-many-locals,too-many-statements
    """Start the daemon collector and serve the dashboard until interrupted."""
    args = parse_arguments()
    scheme = 'https' if args.certfile is not None else 'http'

    devices = Device.all()
    if not devices:
        cprint('ERROR: No NVIDIA devices found.', file=sys.stderr)
        return 1

    devices_info = _describe_devices(devices)

    cprint(
        'INFO: Found {} device(s).'.format(
            colored(str(len(devices)), color='green', attrs=('bold',)),
        ),
        file=sys.stderr,
    )
    for info in devices_info:
        cprint(f'INFO: GPU {info["index"]}: {info["name"]} (UUID: {info["uuid"]})', file=sys.stderr)

    store = MetricStore(retention_seconds=args.retention, interval=args.interval)
    cprint(
        'INFO: Retention {} at {} interval (max {} samples).'.format(
            colored(format_duration(args.retention), color='magenta', attrs=('bold',)),
            colored(f'{args.interval:g}s', color='magenta', attrs=('bold',)),
            colored(str(store.stats()['max_count']), color='magenta', attrs=('bold',)),
        ),
        file=sys.stderr,
    )

    def on_collect(metrics: dict[str, float]) -> bool:
        if store.is_closed():
            return False
        try:
            store.update(metrics)
        except Exception as ex:  # noqa: BLE001 # pylint: disable=broad-except
            message = f'{type(ex).__name__}: {ex}'
            store.record_error(message)
            cprint(f'ERROR: Failed to record metrics sample: {message}', file=sys.stderr)
            return False
        return True

    def on_stop(collector: ResourceMetricCollector) -> None:
        del collector  # suppress unused variable warning
        store.close()

    collector_thread = collect_in_background(
        on_collect,
        ResourceMetricCollector(
            devices,
            root_pids={},  # disable process snapshots
            interval=args.interval,
        ),
        interval=args.interval,
        on_stop=on_stop,
        tag='monitor',
    )

    base_url = f'{scheme}://{args.bind_address}:{args.port}'
    try:
        server = MonitorServer(
            (args.bind_address, args.port),
            store=store,
            devices_info=devices_info,
            hostname=args.hostname,
            interval=args.interval,
        )
    except OSError as ex:
        message = str(ex).lower()
        url_colored = colored(base_url, color='blue', attrs=('bold', 'underline'))
        if 'address already in use' in message:
            cprint(
                f'ERROR: Address {url_colored} is already in use. '
                f'Please specify a different port via `--port <PORT>`.',
                file=sys.stderr,
            )
        elif 'cannot assign requested address' in message:
            cprint(
                f'ERROR: Cannot assign requested address at {url_colored}. '
                f'Please specify a different address via `--bind-address <ADDRESS>`.',
                file=sys.stderr,
            )
        else:
            cprint(f'ERROR: {ex}', file=sys.stderr)
        store.close()
        return 1

    ssl_context = build_ssl_context(args)
    if ssl_context is not None:
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)

    for label, suffix in (
        ('Serving the dashboard at', ''),
        ('  - JSON snapshot:      ', '/metrics.json'),
        ('  - JSON history:       ', '/history.json'),
    ):
        cprint(
            'INFO: {} {}'.format(
                label,
                colored(f'{base_url}{suffix}', color='green', attrs=('bold', 'underline')),
            ),
            file=sys.stderr,
        )

    # Convert SIGTERM into the same KeyboardInterrupt path used by Ctrl-C so containerized
    # / systemd-managed runs (which send SIGTERM, not SIGINT) follow the same graceful path
    # and release the listening socket cleanly.
    def _handle_sigterm(*_args: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, _handle_sigterm)
    join_timeout = max(2.0, args.interval + 1.0)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cprint(file=sys.stderr)
        cprint('INFO: Interrupted by user.', file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        store.close()
        server.server_close()
        collector_thread.join(timeout=join_timeout)
        if collector_thread.is_alive():
            cprint(
                f'WARNING: Collector thread did not stop within {join_timeout:.1f}s; '
                'samples in flight may be lost.',
                file=sys.stderr,
            )

    return 0


if __name__ == '__main__':
    sys.exit(main())
