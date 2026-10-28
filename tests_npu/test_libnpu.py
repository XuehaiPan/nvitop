# -*- coding: utf-8 -*-
"""Unit tests for nvitop.api.libnpu (npu-smi output parsing).

These tests run against real `npu-smi` outputs captured from an Atlas 910B3
server (samples/), so they verify the parsing logic without needing an NPU.
"""

import importlib.util
import pathlib
import sys
import types

# --------------------------------------------------------------------------- #
# Minimal stubs so that `libnpu` can be imported without the full `nvitop`    #
# package (which requires pynvml / psutil).                                   #
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
utils.NA = NA
utils.NaType = NaType

sys.modules['nvitop'] = nvitop
sys.modules['nvitop.api'] = api
sys.modules['nvitop.api.utils'] = utils

_SAMPLES = pathlib.Path(__file__).parent / 'samples'
_LIBNPU = pathlib.Path(__file__).parent.parent / 'nvitop' / 'api' / 'libnpu.py'

spec = importlib.util.spec_from_file_location('nvitop.api.libnpu', _LIBNPU)
libnpu = importlib.util.module_from_spec(spec)
sys.modules['nvitop.api.libnpu'] = libnpu
spec.loader.exec_module(libnpu)  # type: ignore[union-attr]


def load_sample(name):
    return (_SAMPLES / name).read_text(encoding='utf-8')


def test_parse_kv_usages():
    data = libnpu._parse_kv_output(load_sample('info--t-usages--i-0.txt'))  # pylint: disable=protected-access
    assert data['NPU ID'] == '0'
    assert data['HBM Capacity(MB)'] == '65536'
    assert data['HBM Usage Rate(%)'] == '90'
    assert data['Aicore Usage Rate(%)'] == '0'
    assert data['DDR Capacity(MB)'] == '0'


def test_parse_kv_common():
    data = libnpu._parse_kv_output(load_sample('info--t-common--i-0.txt'))  # pylint: disable=protected-access
    assert data['Aicore Freq(MHZ)'] == '1800'
    assert data['Aicore curFreq(MHZ)'] == '800'
    assert data['Aicore Count'] == '20'
    # `Temperature(C)` appears twice (NPU chip then MCU chip); the NPU value wins
    assert int(data['Temperature(C)']) == 33
    assert float(data['NPU Real-time Power(W)']) > 0


def test_parse_kv_temp_power():
    temp = libnpu._parse_kv_output(load_sample('info--t-temp--i-0.txt'))  # pylint: disable=protected-access
    assert int(temp['NPU Temperature (C)']) > 0
    power = libnpu._parse_kv_output(load_sample('info--t-power--i-0.txt'))  # pylint: disable=protected-access
    assert float(power['NPU Real-time Power(W)']) > 0


def test_parse_kv_board():
    board = libnpu._parse_kv_output(load_sample('info--t-board--i-0.txt'))  # pylint: disable=protected-access
    assert board['Product Name'] == 'IT21HMDC_Bin6'
    assert board['Serial Number'] == '1023C7991472'
    assert board['PCIe Bus Info'] == '0000:C1:00.0'
    assert board['Software Version'] == '25.5.1'


def test_parse_kv_memory():
    memory = libnpu._parse_kv_output(load_sample('info--t-memory--i-0.txt'))  # pylint: disable=protected-access
    assert memory['HBM Capacity(MB)'] == '65536'
    assert memory['HBM Clock Speed(MHz)'] == '1600'


def test_proc_mem():
    procs = libnpu._parse_proc_mem(load_sample('info--t-proc-mem--i-0.txt'))  # pylint: disable=protected-access
    assert procs == [{'pid': '3233241', 'name': 'VLLMWorker_TP', 'mem': '55844'}]


def test_device_count():
    assert libnpu._parse_device_count(load_sample('info--m.txt')) == 8  # pylint: disable=protected-access


def test_chip_name():
    assert libnpu._parse_chip_names(load_sample('info--m.txt')) == {  # pylint: disable=protected-access
        0: 'Ascend 910B3',
        1: 'Ascend 910B3',
        2: 'Ascend 910B3',
        3: 'Ascend 910B3',
        4: 'Ascend 910B3',
        5: 'Ascend 910B3',
        6: 'Ascend 910B3',
        7: 'Ascend 910B3',
    }


def test_global():
    result = libnpu._parse_global(load_sample('info.txt'))  # pylint: disable=protected-access
    devices = result['devices']
    assert len(devices) == 8
    d0 = devices[0]
    assert d0['index'] == 0
    assert d0['name'] == '910B3'
    assert d0['health'] == 'OK'
    assert d0['power'] > 0
    assert d0['temperature'] > 0
    assert d0['bus_id'] == '0000:C1:00.0'
    assert d0['aicore'] == 0
    assert d0['memory_used'] > 0
    assert d0['memory_total'] == 65536
    assert d0['ddr_total'] == 0

    procs = result['processes']
    assert len(procs) >= 8  # all 8 cards are occupied by VLLM workers
    p0 = procs[0]
    assert p0['npu'] == 0
    assert p0['chip'] == 0
    assert p0['pid'] == 3233241
    assert p0['name'] == 'VLLMWorker_TP'
    assert p0['mem'] == 55844  # Process memory(MB) is part of the global overview


def test_driver_version():
    assert libnpu._parse_driver_version(load_sample('info.txt')) == '25.5.1'  # pylint: disable=protected-access


def test_forced_global_refresh_populates_shared_cache(monkeypatch):
    calls = []

    def query_raw(*args):
        calls.append(args)
        return load_sample('info.txt')

    libnpu.clear_cache()
    monkeypatch.setattr(libnpu, 'npu_query_raw', query_raw)
    fresh = libnpu.npu_query_global(use_cache=False)
    cached = libnpu.npu_query_global()

    assert len(fresh['devices']) == 8
    assert cached == fresh
    assert calls == [('info',)]


def test_parse_int_float():
    assert libnpu._parse_int('65536') == 65536  # pylint: disable=protected-access
    assert libnpu._parse_int('NA') is libnpu.NA
    assert libnpu._parse_int('-') is libnpu.NA
    assert libnpu._parse_float('99.6') == 99.6  # pylint: disable=protected-access
    assert libnpu._parse_float('NA') is libnpu.NA


def test_compound_field():
    plain, fractions = libnpu._parse_compound_field(  # pylint: disable=protected-access
        '99.5        33                0    / 0',
    )
    assert plain == [99.5, 33]
    assert fractions == [(0, 0)]

    plain, fractions = libnpu._parse_compound_field(  # pylint: disable=protected-access
        '0           0    / 0          59189/ 65536',
    )
    assert plain == [0]
    assert fractions == [(0, 0), (59189, 65536)]


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
