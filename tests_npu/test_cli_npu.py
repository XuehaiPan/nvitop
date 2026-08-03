# -*- coding: utf-8 -*-
"""Unit tests for the ``nvitop-npu`` CLI (rendering and process collection).

The tests load :mod:`nvitop.cli_npu` with stubbed ``libnpu`` / ``npu_device``
modules so they run on any host without an NPU, and verify the rendered
tables, ASCII fallback, terminal-width fitting and process merging logic.
"""

import importlib.util
import pathlib
import re
import sys
import types

# --------------------------------------------------------------------------- #
# Minimal stubs so that `cli_npu` can be imported without pynvml / psutil.     #
# --------------------------------------------------------------------------- #
nvitop = types.ModuleType('nvitop')
nvitop.__path__ = []  # type: ignore[attr-defined]
api = types.ModuleType('nvitop.api')
api.__path__ = []  # type: ignore[attr-defined]
utils = types.ModuleType('nvitop.api.utils')


class NaType(str):
    def __new__(cls, value='N/A'):
        return super().__new__(cls, value)

    def __repr__(self):
        return 'NA'


NA = NaType()


def _colored(text, color=None, attrs=None):
    return str(text)


def _bytes2human(value):
    return f'{value / (1024 ** 2):.0f}MiB'


utils.NA = NA
utils.NaType = NaType
utils.colored = _colored
utils.set_color = lambda value: None
utils.utilization2string = lambda value: 'N/A' if isinstance(value, NaType) else f'{value}%'
utils.bytes2human = _bytes2human

version = types.ModuleType('nvitop.version')
version.__version__ = 'test'

libnpu = types.ModuleType('nvitop.api.libnpu')


class _StubNpuQueryError(Exception):
    pass


class _StubNpuSmiNotFound(Exception):
    pass


libnpu.npu_query_global = lambda *, use_cache=True: {
    'devices': [
        {
            'index': 0,
            'name': '910B3',
            'health': 'OK',
            'power': 99.6,
            'temperature': 33,
            'bus_id': '0000:C1:00.0',
            'aicore': 42,
            'memory_used': 59189,
            'memory_total': 65536,
            'ddr_used': 0,
            'ddr_total': 0,
        },
        {
            'index': 1,
            'name': '910B3',
            'health': 'OK',
            'power': 88.0,
            'temperature': 35,
            'bus_id': '0000:C2:00.0',
            'aicore': 10,
            'memory_used': 12345,
            'memory_total': 65536,
            'ddr_used': 0,
            'ddr_total': 0,
        },
    ],
    'processes': [
        {'npu': 0, 'chip': 0, 'pid': 1001, 'name': 'VLLMWorker_TP', 'mem': 55844},
        {'npu': 1, 'chip': 0, 'pid': 1001, 'name': 'VLLMWorker_TP', 'mem': 1024},
        {'npu': 1, 'chip': 0, 'pid': 1002, 'name': 'python', 'mem': 2048},
    ],
}
libnpu.npu_query_kv_batch = lambda indices, query_type, *, use_cache=True: {
    index: {'Aicore curFreq(MHZ)': '800', 'Aicore Freq(MHZ)': '1800'}
    for index in indices
}


class FakeDevice:
    """Predictable device metrics for CLI tests."""

    def __init__(self, index):
        self.index = index

    @classmethod
    def driver_version(cls):
        return '25.5.1'

    name = lambda self: 'Ascend 910B3'  # noqa: E731
    bus_id = lambda self: f'0000:C{self.index + 1}:00.0'  # noqa: E731
    health = lambda self: 'OK'  # noqa: E731
    gpu_utilization = lambda self: 42 if self.index == 0 else 10  # noqa: E731
    memory_utilization = lambda self: 90 if self.index == 0 else 18  # noqa: E731
    memory_usage = lambda self: '57.80GiB / 64.00GiB'  # noqa: E731
    memory_percent = lambda self: 90.3  # noqa: E731
    power_usage = lambda self: 99600  # noqa: E731
    power_status = lambda self: '99.6W'  # noqa: E731
    temperature = lambda self: 33  # noqa: E731
    aicore_clock = lambda self: '800MHz / 1800MHz'  # noqa: E731
    utilization_color = lambda self, v: 'green'  # noqa: E731
    memory_color = lambda self, v: 'green'  # noqa: E731
    power_color = lambda self, v: 'green'  # noqa: E731
    temperature_color = lambda self, v: 'green'  # noqa: E731
    health_color = lambda self, v: 'green'  # noqa: E731


class StubProcess:
    """Small process object used by collection and rendering tests."""

    def __init__(self, pid, used_memory=None, name=None):
        self.pid = pid
        self.used_memory = used_memory
        self._name = name

    def name(self):
        return self._name or NA

    def username(self):
        return 'root'

    def used_memory_human(self):
        if self.used_memory is None:
            return NA
        return _bytes2human(self.used_memory)


npu_device = types.ModuleType('nvitop.api.npu_device')
npu_device.NpuDevice = FakeDevice
npu_device.NpuProcess = StubProcess
npu_device.NpuQueryError = _StubNpuQueryError
npu_device.NpuSmiNotFound = _StubNpuSmiNotFound

api.libnpu = libnpu
sys.modules['nvitop'] = nvitop
sys.modules['nvitop.api'] = api
sys.modules['nvitop.api.utils'] = utils
sys.modules['nvitop.version'] = version
sys.modules['nvitop.api.libnpu'] = libnpu
sys.modules['nvitop.api.npu_device'] = npu_device

_CLI_PATH = pathlib.Path(__file__).parent.parent / 'nvitop' / 'cli_npu.py'
spec = importlib.util.spec_from_file_location('nvitop.cli_npu_under_test', _CLI_PATH)
cli = importlib.util.module_from_spec(spec)
sys.modules['nvitop.cli_npu_under_test'] = cli
spec.loader.exec_module(cli)


def _visible(text):
    """Strip ANSI escape sequences to measure the visible width of a line."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_border_unicode_and_ascii():
    top = cli._border('┌┬┐', [4, 10], no_unicode=False)  # pylint: disable=protected-access
    assert top == '┌────┬──────────┐'
    ascii_top = cli._border('┌┬┐', [4, 10], no_unicode=True)  # pylint: disable=protected-access
    assert ascii_top == '+----+----------+'
    assert '─' not in ascii_top


def test_vline_switches_with_no_unicode():
    assert cli._vline(no_unicode=False) == '│'  # pylint: disable=protected-access
    assert cli._vline(no_unicode=True) == '|'  # pylint: disable=protected-access


def test_format_cell_truncates_with_ellipsis():
    assert cli._format_cell('Ascend 910B3', 8) == 'Ascend …'  # pylint: disable=protected-access
    assert cli._format_cell('NPU', 10) == 'NPU       '  # pylint: disable=protected-access


def test_fit_columns_drops_tail_columns():
    fitted = cli._fit_columns(cli._DEVICE_COLUMNS, term_width=40)  # pylint: disable=protected-access
    assert len(fitted) < len(cli._DEVICE_COLUMNS)  # pylint: disable=protected-access
    assert fitted[0] == cli._DEVICE_COLUMNS[0]  # NPU column survives first
    total = sum(w for _, w, _ in fitted) + len(fitted) + 1
    assert total <= 40


def test_collect_processes_merges_memory_across_devices():
    processes = cli.collect_processes([FakeDevice(0), FakeDevice(1)], use_cache=False)
    by_pid = {process.pid: (process, npu) for process, npu in processes}
    assert set(by_pid) == {1001, 1002}
    merged, npu = by_pid[1001]
    assert merged.used_memory == (55844 + 1024) * 1024 * 1024  # sum over both devices
    assert npu == 1  # keeps the last device
    _, npu2 = by_pid[1002]
    assert npu2 == 1


def test_collect_processes_filters_devices():
    processes = cli.collect_processes([FakeDevice(0)])
    assert [process.pid for process, _ in processes] == [1001]


def test_filter_processes_by_pid_and_user():
    processes = [(StubProcess(1001, name='a'), 0), (StubProcess(1002, name='b'), 1)]
    filtered = cli.filter_processes(processes, pids={1002})
    assert [p.pid for p, _ in filtered] == [1002]
    filtered = cli.filter_processes(processes, users={'nobody'})
    assert filtered == []


def test_render_devices_table_ascii_has_no_box_drawing():
    table = cli.render_devices_table([FakeDevice(0)], no_unicode=True)
    assert '│' not in table
    assert '|' in table
    assert '+----' in table
    assert '57.80GiB / 64.00GiB' in table
    assert '42%' in table  # AICore utilization of device 0


def test_render_devices_table_unicode():
    table = cli.render_devices_table([FakeDevice(0)])
    assert '┌────┬' in table
    assert '│NPU │' in table


def test_render_processes_table():
    processes = [(StubProcess(1001, used_memory=55844 * 1024 * 1024, name='VLLMWorker_TP'), 0)]
    table = cli.render_processes_table(processes)
    assert 'VLLMWorker_TP' in table
    assert '55844MiB' in table
    assert 'root' in table


def test_render_header_includes_driver_version():
    header = cli.render_header([FakeDevice(0), FakeDevice(1)])
    assert '2 x Ascend 910B3' in header
    assert 'driver 25.5.1' in header


def test_bar_renders_na_and_value():
    assert cli._bar(NA) == cli.EMPTY * cli.BAR_WIDTH  # pylint: disable=protected-access
    filled = cli._bar(50)  # pylint: disable=protected-access
    assert filled.count(cli.BLOCK) == 4  # half of BAR_WIDTH=8


if __name__ == '__main__':
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith('test_')]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as ex:
            failures += 1
            print(f'FAIL {fn.__name__}: {ex}')
        except Exception as ex:  # pylint: disable=broad-except
            failures += 1
            print(f'ERROR {fn.__name__}: {ex!r}')
        else:
            print(f'PASS {fn.__name__}')
    print(f'\n{len(tests) - failures}/{len(tests)} tests passed')
    sys.exit(1 if failures else 0)
