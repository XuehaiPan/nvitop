# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""CUDA visible devices selection tool.

Command line usage:

.. code-block:: bash

    # All devices but sorted
    nvisel       # or use `python3 -m nvitop.select`

    # A simple example to select 4 devices
    nvisel -n 4  # or use `python3 -m nvitop.select -n 4`

    # Select available devices that satisfy the given constraints
    nvisel --min-count 2 --max-count 3 --min-free-memory 5GiB --max-gpu-utilization 60

    # Set `CUDA_VISIBLE_DEVICES` environment variable using `nvisel`
    export CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="$(nvisel -c 1 -f 10GiB)"

    # Use UUID strings in `CUDA_VISIBLE_DEVICES` environment variable
    export CUDA_VISIBLE_DEVICES="$(nvisel -O uuid -c 2 -f 5000M)"

    # Pipe output to other shell utilities
    nvisel -0 -O uuid -c 2 -f 4GiB | xargs -0 -I {} nvidia-smi --id={} --query-gpu=index,memory.free --format=csv

    # Normalize the `CUDA_VISIBLE_DEVICES` environment variable (e.g. convert UUIDs to indices or get full UUIDs for an abbreviated form)
    nvisel -i -S

Python API:

.. code-block:: python

    # Put this at the top of the Python script
    import os
    from nvitop import select_devices

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        select_devices(format='uuid', min_count=4, min_free_memory='8GiB')
    )
"""  # pylint: disable=line-too-long

# pylint: disable=missing-function-docstring

import argparse
import math
import os
import sys
import warnings
from typing import Iterable, List, Optional, Tuple, Union

from nvitop.core import Device, GpuProcess, human2bytes, libnvml
from nvitop.gui import USERNAME, colored
from nvitop.version import __version__


__all__ = ['select_devices']


TTY = sys.stdout.isatty()


# pylint: disable-next=too-many-branches,too-many-statements,too-many-locals
def select_devices(
    devices: Iterable[Device] = None,
    *,
    format: str = 'index',  # pylint: disable=redefined-builtin
    force_index: bool = False,
    min_count: int = 0,
    max_count: Optional[int] = None,
    min_free_memory: Optional[Union[int, str]] = None,  # in bytes or human readable
    min_total_memory: Optional[Union[int, str]] = None,  # in bytes or human readable
    max_gpu_utilization: Optional[int] = None,  # in percentage
    max_memory_utilization: Optional[int] = None,  # in percentage
    tolerance: int = 0,  # in percentage
    free_accounts: List[str] = None,
    sort: bool = True,
    # pylint: disable-next=unused-argument
    **kwargs  # fmt: skip
) -> Union[List[int], List[Tuple[int, int]], List[str]]:
    """Selected a subset of devices satisfying the specified criteria. Returns a list of the device
    identifiers.

    Note:
        The *min count* constraint may not be satisfied if the no enough devices are available. This
        constraint is only enforced when there are both MIG and non-MIG devices present.

    Examples:

        Put the following lines to the top of your script:

        .. code-block:: python

            import os
            from nvitop import select_devices

            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                select_devices(format='uuid', min_count=4, min_free_memory='8GiB')
            )

    Args:
        devices (Iterable[Device]):
            The device superset to select from. If not specified, use all devices as the superset.
        format (str):
            The format of the output. One of :const:`'index'`, :const:`'uuid'`, or :const:`'device'`.
            If gets any MIG device with format :const:`'index'` set, falls back to the :const:`'uuid'`
            format.
        force_index (bool):
            If :data:`True`, always use the device index as the output format when gets any MIG device.
        min_count (int):
            The minimum number of devices to select.
        max_count (Optional[int]):
            The maximum number of devices to select.
        min_free_memory (Optional[Union[int, str]]):
            The minimum free memory (an :class:`int` *in bytes* or a :class:`str` in human readable
            form) of the selected devices.
        min_total_memory (Optional[Union[int, str]]):
            The minimum total memory (an :class:`int` *in bytes* or a :class:`str` in human readable
            form) of the selected devices.
        max_gpu_utilization (Optional[int]):
            The maximum GPU utilization rate (*in percentage*) of the selected devices.
        max_memory_utilization (Optional[int]):
            The maximum memory bandwidth utilization rate (*in percentage*) of the selected devices.
        tolerance (int):
            The tolerance rate (*in percentage*) to loose the constraints.
        free_accounts (List[str]):
            A list of accounts whose used GPU memory needs be considered as free memory.
        sort (bool):
            If :data:`True`, sort the selected devices by memory usage and GPU utilization.
    """

    assert format in ('index', 'uuid', 'device')
    assert tolerance >= 0
    tolerance = tolerance / 100.0

    if max_count is not None:
        if max_count == 0:
            return []
        assert max_count >= min_count >= 0

    free_accounts = set(free_accounts or [])

    if devices is None:
        devices = Device.all()

    if isinstance(min_free_memory, str):
        min_free_memory = human2bytes(min_free_memory)
    if isinstance(min_total_memory, str):
        min_total_memory = human2bytes(min_total_memory)

    available_devices = []  # type: Iterable[DeviceSnapshot]
    for device in devices:
        available_devices.extend(map(lambda device: device.as_snapshot(), device.to_leaf_devices()))
    for device in available_devices:
        device.loosen_constraints = 0

    if len(free_accounts) > 0:
        with GpuProcess.failsafe():
            for device in available_devices:
                as_free_memory = 0
                for process in device.real.processes().values():
                    if process.username() in free_accounts:
                        as_free_memory += process.gpu_memory()
                device.memory_free += as_free_memory
                device.memory_used -= as_free_memory

    if min_free_memory is not None:
        loosen_min_free_memory = min_free_memory * (1.0 - tolerance)
        available_devices = filter(
            lambda device: (
                device.memory_free >= loosen_min_free_memory,
                setattr(
                    device,
                    'loosen_constraints',
                    device.loosen_constraints + int(not device.memory_free >= min_free_memory),
                ),
            )[0],
            available_devices,
        )
    if min_total_memory is not None:
        loosen_min_total_memory = min_total_memory * (1.0 - tolerance)
        available_devices = filter(
            lambda device: (
                device.memory_total >= loosen_min_total_memory,
                setattr(
                    device,
                    'loosen_constraints',
                    device.loosen_constraints + int(not device.memory_total >= min_total_memory),
                ),
            )[0],
            available_devices,
        )
    if max_gpu_utilization is not None:
        loosen_max_gpu_utilization = max_gpu_utilization + 100.0 * tolerance
        available_devices = filter(
            lambda device: (
                device.gpu_utilization <= loosen_max_gpu_utilization,
                setattr(
                    device,
                    'loosen_constraints',
                    device.loosen_constraints
                    + int(not device.gpu_utilization <= max_gpu_utilization),
                ),
            )[0],
            available_devices,
        )
    if max_memory_utilization is not None:
        loosen_max_memory_utilization = max_memory_utilization + 100.0 * tolerance
        available_devices = filter(
            lambda device: (
                device.memory_utilization <= loosen_max_memory_utilization,
                setattr(
                    device,
                    'loosen_constraints',
                    device.loosen_constraints
                    + int(not device.memory_utilization <= max_memory_utilization),
                ),
            )[0],
            available_devices,
        )

    available_devices = list(available_devices)
    if sort:
        available_devices.sort(
            key=lambda device: (
                device.loosen_constraints,
                (not math.isnan(device.memory_free), -device.memory_free),  # descending
                (not math.isnan(device.memory_used), -device.memory_used),  # descending
                (not math.isnan(device.gpu_utilization), device.gpu_utilization),  # ascending
                (not math.isnan(device.memory_utilization), device.memory_utilization),  # ascending
                -device.physical_index,  # descending to keep <GPU 0> free
            )
        )

    if any(device.is_mig_device for device in available_devices):  # found MIG devices!
        non_mig_devices = [device for device in available_devices if not device.is_mig_device]
        mig_devices = [device for device in available_devices if device.is_mig_device]
        if len(non_mig_devices) >= min_count > 0 or not available_devices[0].is_mig_device:
            available_devices = non_mig_devices
        else:
            available_devices = mig_devices[:1]  # at most one MIG device is visible
            if format == 'index' and not force_index:
                format = 'uuid'

    available_devices = available_devices[:max_count]

    if format == 'device':
        return [device.real for device in available_devices]

    if format == 'uuid':
        identifiers = [device.uuid for device in available_devices]  # type: List[str]
    else:
        identifiers = [
            device.index for device in available_devices
        ]  # type: List[int, Tuple[int, int]]
    return identifiers


def parse_arguments():  # pylint: disable=too-many-branches,too-many-statements
    def non_negint(argstring):
        num = int(argstring)
        if num < 0:
            raise ValueError
        return num

    non_negint.__name__ = 'non-negative integer'

    parser = argparse.ArgumentParser(
        prog='nvisel',
        description='CUDA visible devices selection tool.',
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
        version='%(prog)s {}'.format(__version__),
        help="Show %(prog)s's version number and exit.",
    )

    constraints = parser.add_argument_group('constraints')
    constraints.add_argument(
        '--inherit',
        '-i',
        dest='inherit',
        type=str,
        default=argparse.SUPPRESS,
        nargs='?',
        metavar='CUDA_VISIBLE_DEVICES',
        help=(
            'Inherit the given `CUDA_VISIBLE_DEVICES`. If the argument is omitted, use the\n'
            'value from the environment. This means selecting a subset of the currently\n'
            'CUDA-visible devices.'
        ),
    )
    constraints.add_argument(
        '--account-as-free',
        dest='free_accounts',
        nargs='*',
        metavar='USERNAME',
        help=(
            'Account the used GPU memory of the given users as free memory.\n'
            'If this option is specified but without argument, `$USER` will be used.'
        ),
    )
    constraints.add_argument(
        '--min-count',
        '-c',
        dest='min_count',
        type=non_negint,
        default=0,
        metavar='N',
        help=(
            'Minimum number of devices to select. (default: %(default)d)\n'
            'The tool will fail (exit non-zero) if the requested resource is not available.'
        ),
    )
    constraints.add_argument(
        '--max-count',
        '-C',
        dest='max_count',
        type=non_negint,
        default=None,
        metavar='N',
        help='Maximum number of devices to select. (default: all devices)',
    )
    constraints.add_argument(
        '--count',
        '-n',
        dest='count',
        type=non_negint,
        metavar='N',
        help='Overriding both `--min-count N` and `--max-count N`.',
    )
    constraints.add_argument(
        '--min-free-memory',
        '-f',
        dest='min_free_memory',
        type=human2bytes,
        default=None,
        metavar='SIZE',
        help=(
            'Minimum free memory of devices to select. (example value: 4GiB)\n'
            'If this constraint is given, check against all devices.'
        ),
    )
    constraints.add_argument(
        '--min-total-memory',
        '-t',
        dest='min_total_memory',
        type=human2bytes,
        default=None,
        metavar='SIZE',
        help=(
            'Minimum total memory of devices to select. (example value: 10GiB)\n'
            'If this constraint is given, check against all devices.'
        ),
    )
    constraints.add_argument(
        '--max-gpu-utilization',
        '-G',
        dest='max_gpu_utilization',
        type=non_negint,
        default=None,
        metavar='RATE',
        help=(
            'Maximum GPU utilization rate of devices to select. (example value: 30)\n'
            'If this constraint is given, check against all devices.'
        ),
    )
    constraints.add_argument(
        '--max-memory-utilization',
        '-M',
        dest='max_memory_utilization',
        type=non_negint,
        default=None,
        metavar='RATE',
        help=(
            'Maximum memory bandwidth utilization rate of devices to select. (example value: 50)\n'
            'If this constraint is given, check against all devices.'
        ),
    )
    constraints.add_argument(
        '--tolerance',
        '--tol',
        dest='tolerance',
        type=non_negint,
        default=10,
        metavar='TOL',
        help=(
            'The constraints tolerance (in percentage). (default: 0, i.e., strict)\n'
            'This option can loose the constraints if the requested resource is not available.\n'
            'For example, set `--tolerance=20` will accept a device with only 4GiB of free\n'
            'memory when set `--min-free-memory=5GiB`.'
        ),
    )

    formatter = parser.add_argument_group('formatting')
    formatter.add_argument(
        '--format',
        '-O',
        dest='format',
        type=str,
        choices=('index', 'uuid'),
        default='index',
        metavar='FORMAT',
        help=(
            'The output format of the selected device identifiers. (default: %(default)s)\n'
            'If any MIG device found, the output format will be fallback to `uuid`.'
        ),
    )
    separator = formatter.add_mutually_exclusive_group()
    separator.add_argument(
        '--sep',
        '--separator',
        '-s',
        dest='sep',
        type=str,
        default=',',
        metavar='SEP',
        help='Separator for the output. (default: %(default)r)',
    )
    separator.add_argument(
        '--newline',
        dest='newline',
        action='store_true',
        help=r"Use newline character as separator for the output, equivalent to `--sep=$'\n'`.",
    )
    separator.add_argument(
        '--null',
        '-0',
        dest='null',
        action='store_true',
        help=(
            "Use null character ('\\x00') as separator for the output. This option corresponds\n"
            'to the `-0` option of `xargs`.'
        ),
    )
    formatter.add_argument(
        '--no-sort',
        '-S',
        dest='sort',
        action='store_false',
        help='Do not sort the device by memory usage and GPU utilization.',
    )

    args = parser.parse_args()

    if args.count is not None:
        args.min_count = args.max_count = args.count
    if args.max_count is not None and args.max_count < args.min_count:
        raise RuntimeError('Max count must be no less than min count.')

    if args.newline:
        args.sep = '\n'
    elif args.null:
        args.sep = '\0'

    if args.free_accounts is not None and len(args.free_accounts) == 0:
        args.free_accounts.append(USERNAME)

    return args


def main():
    args = parse_arguments()

    try:
        if hasattr(args, 'inherit'):
            if args.inherit is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = args.inherit

            devices = Device.from_cuda_visible_devices()
        else:
            devices = Device.all()
    except libnvml.NVMLError_LibraryNotFound:
        return 1
    except libnvml.NVMLError as ex:
        print(
            '{} {}'.format(colored('NVML ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        return 2
    except RuntimeError as ex:
        print(
            '{} {}'.format(
                colored('CUDA ERROR:', color='red', attrs=('bold',)),
                str(ex).replace('CUDA Error: ', ''),
            ),
            file=sys.stderr,
        )
        return 3

    identifiers = select_devices(devices, **vars(args))
    identifiers = list(map(str, identifiers))
    result = args.sep.join(identifiers)

    if not TTY:
        print('CUDA_VISIBLE_DEVICES="{}"'.format(','.join(identifiers)), file=sys.stderr)

    retval = 0
    if len(identifiers) < args.min_count:
        warnings.warn('Not enough devices found.', RuntimeWarning)
        retval = 4

    if args.sep == '\0':
        print(result, end='\0')
    else:
        print(result)
    return retval


if __name__ == '__main__':
    sys.exit(main())
