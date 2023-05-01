# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2023 Xuehai Pan. All Rights Reserved.
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
"""Prometheus exporter built on top of ``nvitop``."""

from __future__ import annotations

import argparse
import sys

from prometheus_client import start_wsgi_server

import nvitop
from nvitop import Device, colored, libnvml
from nvitop_exporter.exporter import PrometheusExporter
from nvitop_exporter.utils import get_ip_address
from nvitop_exporter.version import __version__


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for ``nvitop-exporter``."""

    def posfloat(argstring: str) -> float:
        num = float(argstring)
        if num <= 0:
            raise ValueError
        return num

    posfloat.__name__ = 'positive float'

    parser = argparse.ArgumentParser(
        prog='nvitop-exporter',
        description='Prometheus exporter built on top of `nvitop`.',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        '--help',
        '-h',
        dest='help',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.',
    )
    parser.add_argument(
        '--version',
        '-V',
        dest='version',
        action='version',
        version=f'%(prog)s {__version__} (nvitop {nvitop.__version__})',
        help="Show %(prog)s's version number and exit.",
    )

    parser.add_argument(
        '--hostname',
        '--host',
        '-H',
        dest='hostname',
        type=str,
        default=get_ip_address(),
        metavar='HOSTNAME',
        help='Hostname to display in the exporter. (default: %(default)s)',
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
        default=8000,
        help='Port to listen on. (default: %(default)d)',
    )
    parser.add_argument(
        '--interval',
        dest='interval',
        type=posfloat,
        default=1.0,
        metavar='SEC',
        help='Interval between updates in seconds. (default: %(default)s)',
    )

    args = parser.parse_args()
    if args.interval < 0.25:
        parser.error(
            f'the interval {args.interval:0.2g}s is too short, which may cause performance issues. '
            f'Expected 1/4 or higher.',
        )

    return args


def main() -> int:  # pylint: disable=too-many-locals,too-many-statements
    """Main function for ``nvitop-exporter`` CLI."""
    args = parse_arguments()

    try:
        device_count = Device.count()
    except libnvml.NVMLError_LibraryNotFound:
        return 1
    except libnvml.NVMLError as ex:
        print(
            '{} {}'.format(colored('NVML ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        return 1

    if device_count == 0:
        print(
            '{} {}'.format(
                colored('NVML ERROR:', color='red', attrs=('bold',)),
                'No NVIDIA devices found.',
            ),
            file=sys.stderr,
        )
        return 1

    devices = Device.from_indices(range(device_count))
    print(
        '{} Found {} NVIDIA device(s).'.format(
            colored('INFO:', color='yellow', attrs=('bold',)),
            colored(str(device_count), color='green', attrs=('bold',)),
        ),
        file=sys.stderr,
    )
    for device in devices:
        print(
            '{} {}'.format(
                colored('INFO:', color='yellow', attrs=('bold',)),
                f'[{device.index}] {device.name()}',
            ),
            file=sys.stderr,
        )

    exporter = PrometheusExporter(devices, hostname=args.hostname, interval=args.interval)

    try:
        start_wsgi_server(port=args.port, addr=args.bind_address)
    except OSError as ex:
        if 'address already in use' in str(ex).lower():
            print(
                '{} {}'.format(
                    colored('ERROR:', color='red', attrs=('bold',)),
                    'Address {} is already in use.'.format(
                        colored(
                            f'http://{args.bind_address}:{args.port}',
                            color='yellow',
                            attrs=('bold', 'underline'),
                        ),
                    ),
                ),
                file=sys.stderr,
            )
        elif 'cannot assign requested address' in str(ex).lower():
            print(
                '{} {}'.format(
                    colored('ERROR:', color='red', attrs=('bold',)),
                    'Cannot assign requested address at {}.'.format(
                        colored(
                            f'http://{args.bind_address}:{args.port}',
                            color='yellow',
                            attrs=('bold', 'underline'),
                        ),
                    ),
                ),
                file=sys.stderr,
            )
        else:
            print(
                '{} {}'.format(
                    colored('ERROR:', color='red', attrs=('bold',)),
                    ex,
                ),
                file=sys.stderr,
            )
        return 1

    print(
        '{} Start the exporter on {} at {}.'.format(
            colored('INFO:', color='yellow', attrs=('bold',)),
            colored(args.hostname, color='magenta', attrs=('bold',)),
            colored(
                f'http://{args.bind_address}:{args.port}/metrics',
                color='green',
                attrs=('bold', 'underline'),
            ),
        ),
        file=sys.stderr,
    )

    try:
        exporter.collect()
    except KeyboardInterrupt:
        print(file=sys.stderr)
        print(
            '{} {}'.format(
                colored('INFO:', color='yellow', attrs=('bold',)),
                'Interrupted by user.',
            ),
            file=sys.stderr,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
