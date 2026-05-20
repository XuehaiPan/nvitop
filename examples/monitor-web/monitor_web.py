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
import ssl
import sys
import threading
import time
import urllib.parse
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

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
    from collections.abc import Sequence


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
    """Print colored ``INFO``/``WARNING``/``ERROR`` lines (mirrors ``nvitop-exporter``)."""
    for prefix, color in (
        ('INFO: ', 'yellow'),
        ('WARNING: ', 'yellow'),
        ('ERROR: ', 'red'),
    ):
        if text.startswith(prefix):
            text = text.replace(
                prefix.rstrip(),
                colored(prefix.rstrip(), color=color, attrs=('bold',)),  # type: ignore[arg-type]
                1,
            )
            break
    print(text, file=file)


class MetricStore:
    """Lock-protected rotating buffer of collector samples.

    Each entry is ``(timestamp, metrics_dict)``.
    The buffer keeps at most ``int(retention / interval)`` samples; older entries are evicted
    automatically by :class:`deque`.
    """

    def __init__(self, *, retention_seconds: float, interval: float) -> None:
        """Build an empty buffer sized for ``retention_seconds`` at ``interval`` per sample."""
        maxlen = max(1, int(retention_seconds / interval))
        self._lock = threading.Lock()
        self._retention_seconds = retention_seconds
        self._samples: deque[tuple[float, dict[str, float]]] = deque(maxlen=maxlen)
        self._closed = False

    def update(self, metrics: dict[str, float]) -> None:
        """Append one sample; oldest entries are evicted by the deque ``maxlen``."""
        sample = (time.time(), dict(metrics))
        with self._lock:
            self._samples.append(sample)

    def latest(self) -> tuple[float, dict[str, float]] | None:
        """Return the most recent sample, or :data:`None` if the buffer is empty."""
        with self._lock:
            return self._samples[-1] if self._samples else None

    def history(
        self,
        *,
        limit: int | None = None,
        since: float | None = None,
    ) -> list[tuple[float, dict[str, float]]]:
        """Snapshot copy of the buffer, optionally trimmed by ``limit`` and ``since``."""
        with self._lock:
            samples = list(self._samples)
        if since is not None:
            samples = [s for s in samples if s[0] > since]
        if limit is not None and len(samples) > limit:
            samples = samples[-limit:]
        return samples

    def stats(self) -> dict[str, Any]:
        """Return buffer statistics suitable for embedding in the JSON payload."""
        with self._lock:
            count = len(self._samples)
            oldest = self._samples[0][0] if count else 0.0
            newest = self._samples[-1][0] if count else 0.0
            max_count = self._samples.maxlen or 0
        return {
            'count': count,
            'max_count': max_count,
            'retention_seconds': self._retention_seconds,
            'retention_human': format_duration(self._retention_seconds),
            'oldest_epoch': oldest,
            'newest_epoch': newest,
        }

    def close(self) -> None:
        """Mark the store closed so the next ``on_collect`` callback returns :data:`False`."""
        with self._lock:
            self._closed = True

    def is_closed(self) -> bool:
        """Return :data:`True` once :meth:`close` has been called."""
        with self._lock:
            return self._closed


HTML_PATH = Path(__file__).resolve().with_suffix('.html')


class MonitorRequestHandler(http.server.BaseHTTPRequestHandler):
    """Tiny request router serving the dashboard HTML and JSON snapshots."""

    server_version = 'nvitop-monitor-web'
    sys_version = ''

    store: ClassVar[MetricStore]  # populated in main() before serve_forever
    devices_info: ClassVar[list[dict[str, Any]]] = []
    interval: ClassVar[float] = 1.0

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

    def _send_html(self) -> None:
        body = HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_metrics_json(self) -> None:
        latest = self.store.latest()
        sample_time = latest[0] if latest is not None else 0.0
        metrics = latest[1] if latest is not None else {}
        now = time.time()
        payload = {
            'interval': self.interval,
            'server_time': now,
            'sample_time': sample_time,
            'stale_seconds': max(0.0, now - sample_time) if latest is not None else None,
            'buffer': self.store.stats(),
            'devices': self.devices_info,
            'metrics': metrics,
            'metrics_human': _humanize_metrics(metrics),
        }
        self._send_json(payload)

    def _send_history_json(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        limit = _maybe_positive_int(params.get('limit', [None])[0])
        since = _maybe_float(params.get('since', [None])[0])
        history = self.store.history(limit=limit, since=since)
        payload = {
            'buffer': self.store.stats(),
            'samples': [{'epoch': ts, 'metrics': metrics} for ts, metrics in history],
        }
        self._send_json(payload)

    def _send_json(self, payload: object) -> None:
        # `allow_nan=False` makes strict JSON; ``_finite()`` first maps `math.nan`/`math.inf` (which
        # the collector emits for missing samples) to `None` so the browser's `JSON.parse` accepts
        # the body.
        body = json.dumps(_finite(payload), allow_nan=False, default=float).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self) -> None:
        body = b'404 Not Found\n'
        self.send_response(404)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _finite(value: Any) -> Any:
    """Replace `nan`/`+inf`/`-inf` with :data:`None` so the result is strict JSON.

    The collector writes :data:`math.nan` for any metric key that was sampled previously but is
    missing from the current snapshot (see :class:`nvitop.ResourceMetricCollector`), and strict
    JSON has no representation for ``NaN`` or ``Infinity``.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


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


def _maybe_positive_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _maybe_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def build_ssl_context(args: argparse.Namespace) -> ssl.SSLContext | None:
    """Build an :class:`ssl.SSLContext` from the parsed args, or :data:`None` for plain HTTP."""
    if args.certfile is None and args.keyfile is None:
        return None
    # `parse_arguments()` already enforces that both flags are set together.
    assert args.certfile is not None
    assert args.keyfile is not None
    ctx = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile=args.certfile, keyfile=args.keyfile)
    if args.client_cafile is not None or args.client_capath is not None:
        ctx.load_verify_locations(cafile=args.client_cafile, capath=args.client_capath)
        ctx.verify_mode = ssl.CERT_REQUIRED if args.client_auth_required else ssl.CERT_OPTIONAL
    return ctx


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the web monitor."""

    def posfloat(arg: str) -> float:
        value = float(arg)
        if value <= 0:
            raise ValueError
        return value

    posfloat.__name__ = 'positive float'

    parser = argparse.ArgumentParser(
        prog='monitor_web.py',
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
            'Requires `--client-auth-required` to actually verify client certificates.'
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
            'Requires `--client-auth-required` to actually verify client certificates.'
        ),
    )
    tls.add_argument(
        '--client-auth-required',
        dest='client_auth_required',
        action='store_true',
        help=(
            'Require clients to present a valid certificate (mutual TLS).\n'
            'Requires `--client-cafile` or `--client-capath`.'
        ),
    )

    args = parser.parse_args()

    if args.interval < _MIN_INTERVAL:
        parser.error(
            f'the interval {args.interval:0.2g}s is too short, '
            f'which may cause performance issues. Expected 1/4 or higher.',
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
        cprint(
            f'INFO: GPU {info["index"]}: {info["name"]} (UUID: {info["uuid"]})',
            file=sys.stderr,
        )

    store = MetricStore(retention_seconds=args.retention, interval=args.interval)
    cprint(
        'INFO: Retention {} at {} interval (max {} samples).'.format(
            colored(
                format_duration(args.retention),
                color='magenta',
                attrs=('bold',),
            ),
            colored(
                f'{args.interval:g}s',
                color='magenta',
                attrs=('bold',),
            ),
            colored(
                str(store.stats()['max_count']),
                color='magenta',
                attrs=('bold',),
            ),
        ),
        file=sys.stderr,
    )

    def on_collect(metrics: dict[str, float]) -> bool:
        if store.is_closed():
            return False
        store.update(metrics)
        return True

    def on_stop(collector: ResourceMetricCollector) -> None:
        del collector  # suppress unused variable warning
        store.close()

    collect_in_background(
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

    MonitorRequestHandler.store = store
    MonitorRequestHandler.devices_info = devices_info
    MonitorRequestHandler.interval = args.interval

    base_url = f'{scheme}://{args.bind_address}:{args.port}'
    try:
        server = http.server.ThreadingHTTPServer(
            (args.bind_address, args.port),
            MonitorRequestHandler,
        )
    except OSError as ex:
        message = str(ex).lower()
        url_colored = colored(
            base_url,
            color='blue',
            attrs=('bold', 'underline'),
        )
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
                colored(
                    f'{base_url}{suffix}',
                    color='green',
                    attrs=('bold', 'underline'),
                ),
            ),
            file=sys.stderr,
        )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cprint(file=sys.stderr)
        cprint('INFO: Interrupted by user.', file=sys.stderr)
    finally:
        store.close()
        server.shutdown()
        server.server_close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
