# -*- coding: utf-8 -*-
"""Unit tests for the standalone Ascend NPU monitor UI."""

import importlib.util
import os
import pathlib
import re
import sys
import types


class NaType(str):
    """Minimal stand-in for :class:`nvitop.api.utils.NaType`."""


NA = NaType('N/A')


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
        return f'{self.used_memory // (1024 * 1024)}MiB'


class StubDeviceType:
    """Placeholder imported by the CLI module."""


class NpuQueryError(Exception):
    """Placeholder query error."""


class NpuSmiNotFound(Exception):
    """Placeholder executable error."""


def _colored(text, color=None, attrs=None):
    del color, attrs
    return str(text)


def _bytes2human(value):
    return f'{value / (1024**3):.1f}GiB'


nvitop = types.ModuleType('nvitop')
nvitop.__path__ = []
api = types.ModuleType('nvitop.api')
api.__path__ = []
libnpu = types.ModuleType('nvitop.api.libnpu')
libnpu.npu_query_global = lambda: {'devices': []}
libnpu.npu_query_proc_mem = lambda index, use_cache=True: []
npu_device = types.ModuleType('nvitop.api.npu_device')
npu_device.NpuDevice = StubDeviceType
npu_device.NpuProcess = StubProcess
npu_device.NpuQueryError = NpuQueryError
npu_device.NpuSmiNotFound = NpuSmiNotFound
utils = types.ModuleType('nvitop.api.utils')
utils.NA = NA
utils.NaType = NaType
utils.bytes2human = _bytes2human
utils.colored = _colored
utils.set_color = lambda value: None
utils.utilization2string = lambda value: 'N/A' if isinstance(value, NaType) else f'{value}%'
version = types.ModuleType('nvitop.version')
version.__version__ = 'test'
api.libnpu = libnpu

sys.modules['nvitop'] = nvitop
sys.modules['nvitop.api'] = api
sys.modules['nvitop.api.libnpu'] = libnpu
sys.modules['nvitop.api.npu_device'] = npu_device
sys.modules['nvitop.api.utils'] = utils
sys.modules['nvitop.version'] = version

_CLI_PATH = pathlib.Path(__file__).parent.parent / 'nvitop' / 'cli_npu.py'
spec = importlib.util.spec_from_file_location('nvitop.cli_npu_under_test', _CLI_PATH)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class FakeDevice:
    """Predictable device metrics for dashboard tests."""

    def __init__(self, index, *, gpu, used_gib, total_gib, temperature, power, health='OK'):
        self.index = index
        self._gpu = gpu
        self._used = used_gib * 1024**3
        self._total = total_gib * 1024**3
        self._temperature = temperature
        self._power = power * 1000
        self._health = health
        self.refreshed_with = None

    def refresh(self, global_info=None):
        self.refreshed_with = global_info

    def name(self):
        return 'Ascend 910B3'

    def bus_id(self):
        return f'0000:{self.index:02X}:00.0'

    def health(self):
        return self._health

    def health_color(self, value):
        return 'green' if value == 'OK' else 'red'

    def gpu_utilization(self):
        return self._gpu

    def utilization_color(self, value):
        del value
        return 'green'

    def memory_used(self):
        return self._used

    def memory_total(self):
        return self._total

    def memory_percent(self):
        return self._used / self._total * 100

    def memory_color(self, value):
        del value
        return 'yellow'

    def memory_usage(self):
        return f'{self._used // 1024**3}GiB / {self._total // 1024**3}GiB'

    def power_usage(self):
        return self._power

    def power_status(self):
        return f'{self._power / 1000:.1f}W'

    def power_color(self, value):
        del value
        return 'green'

    def temperature(self):
        return self._temperature

    def temperature_color(self, value):
        del value
        return 'green'

    def aicore_clock(self):
        return 800


def _visible(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def test_format_cell_measures_colored_text_by_visible_width():
    value = '\x1b[31mOK\x1b[0m'
    formatted = cli._format_cell(value, 7)
    assert formatted.startswith('\x1b[31mOK\x1b[0m')
    assert len(_visible(formatted)) == 7


def test_collect_processes_merges_memory_and_keeps_all_npus(monkeypatch):
    rows = {
        0: [{'pid': '42', 'name': 'worker', 'mem': '100'}],
        1: [{'pid': '42', 'name': 'worker', 'mem': '200'}],
    }
    monkeypatch.setattr(cli.libnpu, 'npu_query_proc_mem', lambda index, use_cache=True: rows[index])

    [(process, npus)] = cli.collect_processes([types.SimpleNamespace(index=0), types.SimpleNamespace(index=1)])

    assert process.pid == 42
    assert process.used_memory == 300 * 1024 * 1024
    assert npus == (0, 1)


def test_refresh_devices_uses_one_global_query(monkeypatch):
    devices = [
        FakeDevice(0, gpu=20, used_gib=32, total_gib=64, temperature=50, power=100),
        FakeDevice(1, gpu=80, used_gib=64, total_gib=64, temperature=70, power=200),
    ]
    calls = []

    def query_global():
        calls.append(True)
        return {'devices': [{'index': 0, 'health': 'OK'}, {'index': 1, 'health': 'OK'}]}

    monkeypatch.setattr(cli.libnpu, 'npu_query_global', query_global)
    cli.refresh_devices(devices)

    assert len(calls) == 1
    assert devices[0].refreshed_with == {'index': 0, 'health': 'OK'}
    assert devices[1].refreshed_with == {'index': 1, 'health': 'OK'}


def test_render_summary_aggregates_cluster_metrics():
    devices = [
        FakeDevice(0, gpu=20, used_gib=32, total_gib=64, temperature=50, power=100),
        FakeDevice(1, gpu=80, used_gib=64, total_gib=64, temperature=70, power=200),
    ]

    output = cli.render_summary(devices, process_count=2, no_unicode=True, width=120)

    assert '2/2 OK' in output
    assert 'AICore 50%' in output
    assert 'HBM 75%' in output
    assert 'Power 300W' in output
    assert 'Peak 70C' in output
    assert 'Processes 2' in output
    assert all(len(_visible(line)) <= 120 for line in output.splitlines())


def test_render_devices_table_shows_utilization_bars(monkeypatch):
    device = FakeDevice(0, gpu=50, used_gib=32, total_gib=64, temperature=50, power=100)
    monkeypatch.setattr(cli.shutil, 'get_terminal_size', lambda: os.terminal_size((120, 40)))

    output = cli.render_devices_table([device], no_unicode=True)

    assert '████░░░░ 50%' in output
    assert '████░░░░ 50%' in output
    assert all(len(_visible(line)) <= 120 for line in output.splitlines())
