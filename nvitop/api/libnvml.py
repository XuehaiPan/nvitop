# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2024 Xuehai Pan. All Rights Reserved.
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
"""Utilities for the NVML Python bindings (`nvidia-ml-py <https://pypi.org/project/nvidia-ml-py>`_)."""

# pylint: disable=invalid-name

from __future__ import annotations

import atexit as _atexit
import ctypes as _ctypes
import inspect as _inspect
import logging as _logging
import os as _os
import re as _re
import sys as _sys
import threading as _threading
import time as _time
from types import FunctionType as _FunctionType
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any
from typing import Callable as _Callable
from typing import ClassVar as _ClassVar

# Python Bindings for the NVIDIA Management Library (NVML)
# https://pypi.org/project/nvidia-ml-py
import pynvml as _pynvml
from pynvml import *  # noqa: F403 # pylint: disable=wildcard-import,unused-wildcard-import
from pynvml import nvmlDeviceGetPciInfo  # appease mypy # noqa: F401 # pylint: disable=unused-import

from nvitop.api.utils import NA, UINT_MAX, ULONGLONG_MAX, NaType
from nvitop.api.utils import colored as __colored


if _TYPE_CHECKING:
    from typing_extensions import Self as _Self  # Python 3.11+
    from typing_extensions import TypeAlias as _TypeAlias  # Python 3.10+


__all__ = [  # will be updated in below
    'NA',
    'UINT_MAX',
    'ULONGLONG_MAX',
    'nvmlCheckReturn',
    'nvmlQuery',
    'nvmlQueryFieldValues',
    'nvmlInit',
    'nvmlInitWithFlags',
    'nvmlShutdown',
    'NVMLError',
]


if not callable(getattr(_pynvml, 'nvmlInitWithFlags', None)):
    raise ImportError(  # noqa: TRY004
        'Your installed package `nvidia-ml-py` is corrupted. Please reinstall package '
        '`nvidia-ml-py` via `pip3 install --force-reinstall nvidia-ml-py nvitop`.',
    )


# Members from `pynvml` ############################################################################

NVMLError: type[_pynvml.NVMLError] = _pynvml.NVMLError
NVMLError.__doc__ = """Base exception class for NVML query errors."""
NVMLError.__new__.__doc__ = """Map value to a proper subclass of :class:`NVMLError`."""
nvmlExceptionClass: _Callable[[int], type[_pynvml.NVMLError]] = _pynvml.nvmlExceptionClass
nvmlExceptionClass.__doc__ = """Map value to a proper subclass of :class:`NVMLError`."""

# Load members from module `pynvml` and register them in `__all__` and globals.
_vars_pynvml = vars(_pynvml)
_name = _attr = None
_errcode_to_name = {}
_const_names = []
_errcode_to_string = NVMLError._errcode_to_string  # pylint: disable=protected-access

# 1. Put error classes in `__all__` first
for _name, _attr in _vars_pynvml.items():
    if _name in {'nvmlInit', 'nvmlInitWithFlags', 'nvmlShutdown'}:
        continue
    if _name.startswith(('NVML_ERROR_', 'NVMLError_')):
        __all__.append(_name)  # noqa: PYI056
        if _name.startswith('NVML_ERROR_'):
            _errcode_to_name[_attr] = _name
            _const_names.append(_name)

# 2. Then the remaining members
for _name, _attr in _vars_pynvml.items():
    if _name in {'nvmlInit', 'nvmlInitWithFlags', 'nvmlShutdown'}:
        continue
    if (_name.startswith('NVML_') and not _name.startswith('NVML_ERROR_')) or (
        _name.startswith('nvml') and isinstance(_attr, _FunctionType)
    ):
        __all__.append(_name)  # noqa: PYI056
        if _name.startswith('NVML_'):
            _const_names.append(_name)

# 3. Add docstring to exception classes
_errcode = _reason = _subclass = None
for _errcode, _reason in _errcode_to_string.items():
    _subclass = nvmlExceptionClass(_errcode)
    _subclass.__doc__ = '{}. Code: :data:`{}` ({})'.format(
        _reason.rstrip('.'),
        _errcode_to_name[_errcode],
        _errcode,
    )

# 4. Add undocumented constants into module docstring
_data_docs = []
_sphinx_doc = None
for _name in _const_names:
    _attr = _vars_pynvml[_name]
    _sphinx_doc = f"""
.. data:: {_name}
    :type: {_attr.__class__.__name__}
    :value: {_attr!r}
"""
    if _name.startswith('NVML_ERROR_') and _attr in _errcode_to_string:
        _reason = _errcode_to_string[_attr]
        _sphinx_doc += """
    {}. See also class :class:`NVMLError` and :class:`{}`.
""".format(_reason.rstrip('.'), nvmlExceptionClass(_attr).__name__)  # fmt: skip
    _data_docs.append(_sphinx_doc.strip())
__doc__ += """

---------

Constants
^^^^^^^^^

{}

---------

Functions and Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: __enter__() -> libnvml

    Entry of the context manager for ``with`` statement.

.. function:: __exit__(*args, **kwargs) -> None

    Shutdown the NVML context in the context manager for ``with`` statement.

""".format('\n\n'.join(_data_docs))  # fmt: skip

del (
    _name,
    _attr,
    _vars_pynvml,
    _errcode,
    _reason,
    _subclass,
    _errcode_to_name,
    _errcode_to_string,
    _const_names,
    _data_docs,
    _sphinx_doc,
)

# 5. Add explicit references to appease linters
# pylint: disable=no-member
c_nvmlDevice_t: _TypeAlias = _pynvml.c_nvmlDevice_t  # noqa: PYI042
c_nvmlFieldValue_t: _TypeAlias = _pynvml.c_nvmlFieldValue_t  # noqa: PYI042
NVML_SUCCESS: int = _pynvml.NVML_SUCCESS
NVML_ERROR_INSUFFICIENT_SIZE: int = _pynvml.NVML_ERROR_INSUFFICIENT_SIZE
NVMLError_FunctionNotFound: _TypeAlias = _pynvml.NVMLError_FunctionNotFound
NVMLError_GpuIsLost: _TypeAlias = _pynvml.NVMLError_GpuIsLost
NVMLError_InvalidArgument: _TypeAlias = _pynvml.NVMLError_InvalidArgument
NVMLError_LibraryNotFound: _TypeAlias = _pynvml.NVMLError_LibraryNotFound
NVMLError_NoPermission: _TypeAlias = _pynvml.NVMLError_NoPermission
NVMLError_NotFound: _TypeAlias = _pynvml.NVMLError_NotFound
NVMLError_NotSupported: _TypeAlias = _pynvml.NVMLError_NotSupported
NVMLError_Unknown: _TypeAlias = _pynvml.NVMLError_Unknown
NVML_CLOCK_GRAPHICS: int = _pynvml.NVML_CLOCK_GRAPHICS
NVML_CLOCK_SM: int = _pynvml.NVML_CLOCK_SM
NVML_CLOCK_MEM: int = _pynvml.NVML_CLOCK_MEM
NVML_CLOCK_VIDEO: int = _pynvml.NVML_CLOCK_VIDEO
NVML_TEMPERATURE_GPU: int = _pynvml.NVML_TEMPERATURE_GPU
NVML_DRIVER_WDDM: int = _pynvml.NVML_DRIVER_WDDM
NVML_DRIVER_WDM: int = _pynvml.NVML_DRIVER_WDM
NVML_MEMORY_ERROR_TYPE_UNCORRECTED: int = _pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED
NVML_VOLATILE_ECC: int = _pynvml.NVML_VOLATILE_ECC
NVML_COMPUTEMODE_DEFAULT: int = _pynvml.NVML_COMPUTEMODE_DEFAULT
NVML_COMPUTEMODE_EXCLUSIVE_THREAD: int = _pynvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD
NVML_COMPUTEMODE_PROHIBITED: int = _pynvml.NVML_COMPUTEMODE_PROHIBITED
NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: int = _pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS
NVML_PCIE_UTIL_TX_BYTES: int = _pynvml.NVML_PCIE_UTIL_TX_BYTES
NVML_PCIE_UTIL_RX_BYTES: int = _pynvml.NVML_PCIE_UTIL_RX_BYTES
NVML_NVLINK_MAX_LINKS: int = _pynvml.NVML_NVLINK_MAX_LINKS
NVML_FI_DEV_NVLINK_LINK_COUNT: int = _pynvml.NVML_FI_DEV_NVLINK_LINK_COUNT
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX: int = _pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX: int = _pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX
NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX: int = _pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX
NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX: int = _pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX
NVML_VALUE_TYPE_DOUBLE: int = getattr(_pynvml, 'NVML_VALUE_TYPE_DOUBLE', 0)
NVML_VALUE_TYPE_UNSIGNED_INT: int = getattr(_pynvml, 'NVML_VALUE_TYPE_UNSIGNED_INT', 1)
NVML_VALUE_TYPE_UNSIGNED_LONG: int = getattr(_pynvml, 'NVML_VALUE_TYPE_UNSIGNED_LONG', 2)
NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: int = getattr(_pynvml, 'NVML_VALUE_TYPE_UNSIGNED_LONG_LONG', 3)
NVML_VALUE_TYPE_SIGNED_LONG_LONG: int = getattr(_pynvml, 'NVML_VALUE_TYPE_SIGNED_LONG_LONG', 4)
NVML_VALUE_TYPE_SIGNED_INT: int = getattr(_pynvml, 'NVML_VALUE_TYPE_SIGNED_INT', 5)
# pylint: enable=no-member

# New members in `libnvml` #########################################################################

__flags: list[int] = []
__initialized: bool = False
__lock: _threading.Lock = _threading.Lock()

LOGGER: _logging.Logger = _logging.getLogger(__name__)
try:
    LOGGER.setLevel(_os.getenv('LOGLEVEL', default='WARNING').upper())
except (ValueError, TypeError):
    pass
if not LOGGER.hasHandlers() and LOGGER.isEnabledFor(_logging.DEBUG):
    _formatter = _logging.Formatter(
        '[%(levelname)s] %(asctime)s %(name)s::%(funcName)s: %(message)s',
    )
    _stream_handler = _logging.StreamHandler()
    _stream_handler.setFormatter(_formatter)
    _file_handler = _logging.FileHandler('nvitop.log')
    _file_handler.setFormatter(_formatter)
    LOGGER.addHandler(_stream_handler)
    LOGGER.addHandler(_file_handler)
    del _formatter, _stream_handler, _file_handler

UNKNOWN_FUNCTIONS: dict[str, tuple[_Callable | str, NVMLError_FunctionNotFound]] = {}
UNKNOWN_FUNCTIONS_CACHE_SIZE: int = 1024
VERSIONED_PATTERN: _re.Pattern = _re.compile(r'^(?P<name>\w+)(?P<suffix>_v(\d)+)$')


def _lazy_init() -> None:
    """Lazily initialize the NVML context.

    Raises:
        NVMLError_LibraryNotFound:
            If cannot find the NVML library, usually the NVIDIA driver is not installed.
        NVMLError_DriverNotLoaded:
            If NVIDIA driver is not loaded.
        NVMLError_LibRmVersionMismatch:
            If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
            driver without reloading the kernel module.
        AttributeError:
            If cannot find function :func:`pynvml.nvmlInitWithFlags`, usually the :mod:`pynvml` module
            is overridden by other modules. Need to reinstall package ``nvidia-ml-py``.
    """
    with __lock:
        if __initialized:
            return
    nvmlInit()
    _atexit.register(nvmlShutdown)


def nvmlInit() -> None:  # pylint: disable=function-redefined
    """Initialize the NVML context with default flag (0).

    Raises:
        NVMLError_LibraryNotFound:
            If cannot find the NVML library, usually the NVIDIA driver is not installed.
        NVMLError_DriverNotLoaded:
            If NVIDIA driver is not loaded.
        NVMLError_LibRmVersionMismatch:
            If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
            driver without reloading the kernel module.
        AttributeError:
            If cannot find function :func:`pynvml.nvmlInitWithFlags`, usually the :mod:`pynvml` module
            is overridden by other modules. Need to reinstall package ``nvidia-ml-py``.
    """
    nvmlInitWithFlags(0)


def nvmlInitWithFlags(flags: int) -> None:  # pylint: disable=function-redefined
    """Initialize the NVML context with the given flags.

    Raises:
        NVMLError_LibraryNotFound:
            If cannot find the NVML library, usually the NVIDIA driver is not installed.
        NVMLError_DriverNotLoaded:
            If NVIDIA driver is not loaded.
        NVMLError_LibRmVersionMismatch:
            If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
            driver without reloading the kernel module.
        AttributeError:
            If cannot find function :func:`pynvml.nvmlInitWithFlags`, usually the :mod:`pynvml` module
            is overridden by other modules. Need to reinstall package ``nvidia-ml-py``.
    """
    global __flags, __initialized  # pylint: disable=global-statement,global-variable-not-assigned

    with __lock:
        if len(__flags) > 0 and flags == __flags[-1]:
            __initialized = True
            return

    try:
        _pynvml.nvmlInitWithFlags(flags)
    except NVMLError_LibraryNotFound:
        message = (
            'FATAL ERROR: NVIDIA Management Library (NVML) not found.\n'
            'HINT: The NVIDIA Management Library ships with the NVIDIA display driver (available at\n'
            '      https://www.nvidia.com/Download/index.aspx), or can be downloaded as part of the\n'
            '      NVIDIA CUDA Toolkit (available at https://developer.nvidia.com/cuda-downloads).\n'
            '      The lists of OS platforms and NVIDIA-GPUs supported by the NVML library can be\n'
            '      found in the NVML API Reference at https://docs.nvidia.com/deploy/nvml-api.'
        )
        for text, color, attrs in (
            ('FATAL ERROR:', 'red', ('bold',)),
            ('HINT:', 'yellow', ('bold',)),
            ('https://www.nvidia.com/Download/index.aspx', None, ('underline',)),
            ('https://developer.nvidia.com/cuda-downloads', None, ('underline',)),
            ('https://docs.nvidia.com/deploy/nvml-api', None, ('underline',)),
        ):
            message = message.replace(text, __colored(text, color=color, attrs=attrs))

        LOGGER.critical(message)
        raise
    except AttributeError:
        message = (
            'FATAL ERROR: The dependency package `nvidia-ml-py` is corrupted. You may have installed\n'
            '             other packages overriding the module `pynvml`.\n'
            'Please reinstall `nvitop` with command:\n'
            '    python3 -m pip install --force-reinstall nvitop'
        )
        for text, color, attrs in (
            ('FATAL ERROR:', 'red', ('bold',)),
            ('nvidia-ml-py', None, ('bold',)),
            ('pynvml', None, ('bold',)),
            ('nvitop', None, ('bold',)),
        ):
            message = message.replace(text, __colored(text, color=color, attrs=attrs), 1)

        LOGGER.critical(message)
        raise

    with __lock:
        __flags.append(flags)
        __initialized = True


def nvmlShutdown() -> None:  # pylint: disable=function-redefined
    """Shutdown the NVML context.

    Raises:
        NVMLError_LibraryNotFound:
            If cannot find the NVML library, usually the NVIDIA driver is not installed.
        NVMLError_DriverNotLoaded:
            If NVIDIA driver is not loaded.
        NVMLError_LibRmVersionMismatch:
            If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
            driver without reloading the kernel module.
        NVMLError_Uninitialized:
            If NVML was not first initialized with :func:`nvmlInit`.
    """
    global __flags, __initialized  # pylint: disable=global-statement,global-variable-not-assigned

    _pynvml.nvmlShutdown()
    with __lock:
        try:
            __flags.pop()
        except IndexError:
            pass
        __initialized = len(__flags) > 0


def nvmlQuery(
    func: _Callable[..., _Any] | str,
    *args: _Any,
    default: _Any = NA,
    ignore_errors: bool = True,
    ignore_function_not_found: bool = False,
    **kwargs: _Any,
) -> _Any:
    """Call a function with the given arguments from NVML.

    The NVML context will be automatically initialized.

    Args:
        func (Union[Callable[..., Any], str]):
            The function to call. If it is given by string, lookup for the function first from
            module :mod:`pynvml`.
        default (Any):
            The default value if the query fails.
        ignore_errors (bool):
            Whether to ignore errors and return the default value.
        ignore_function_not_found (bool):
            Whether to ignore function not found errors and return the default value. If set to
            :data:`False`, an error message will be logged to the logger.
        *args:
            Positional arguments to pass to the query function.
        **kwargs:
            Keyword arguments to pass to the query function.

    Raises:
        NVMLError_LibraryNotFound:
            If cannot find the NVML library, usually the NVIDIA driver is not installed.
        NVMLError_DriverNotLoaded:
            If NVIDIA driver is not loaded.
        NVMLError_LibRmVersionMismatch:
            If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
            driver without reloading the kernel module.
        NVMLError_FunctionNotFound:
            If the function is not found, usually the installed ``nvidia-ml-py`` is not compatible
            with the installed NVIDIA driver.
        NVMLError_NotSupported:
            If the function is not supported by the driver or the device.
        NVMLError_InvalidArgument:
            If passed with an invalid argument.
    """
    global UNKNOWN_FUNCTIONS  # pylint: disable=global-statement,global-variable-not-assigned

    _lazy_init()

    try:
        if isinstance(func, str):
            try:
                func = getattr(__modself, func)
            except AttributeError as e1:
                raise NVMLError_FunctionNotFound from e1

        try:
            retval = func(*args, **kwargs)  # type: ignore[operator]
        except UnicodeDecodeError as e2:
            raise NVMLError_Unknown from e2
    except NVMLError_FunctionNotFound as e3:
        if not ignore_function_not_found:
            identifier = (
                func
                if isinstance(func, str)
                else (_inspect.getsource(func) if func.__name__ == '<lambda>' else repr(func))
            )
            with __lock:
                if (
                    identifier not in UNKNOWN_FUNCTIONS
                    and len(UNKNOWN_FUNCTIONS) < UNKNOWN_FUNCTIONS_CACHE_SIZE
                ):
                    UNKNOWN_FUNCTIONS[identifier] = (func, e3)
                    LOGGER.exception(
                        (
                            'ERROR: A FunctionNotFound error occurred while calling %s.\n'
                            'Please verify whether the `nvidia-ml-py` package is '
                            'compatible with your NVIDIA driver version.'
                        ),
                        f'nvmlQuery({func!r}, *args, **kwargs)',
                    )
        if ignore_errors or ignore_function_not_found:
            return default
        raise
    except NVMLError:
        if ignore_errors:
            return default
        raise

    if isinstance(retval, bytes):
        retval = retval.decode('utf-8', errors='replace')
    return retval


def nvmlQueryFieldValues(
    handle: c_nvmlDevice_t,
    field_ids: list[int | tuple[int, int]],
) -> list[tuple[float | int | NaType, int]]:
    """Query multiple field values from NVML.

    Request values for a list of fields for a device. This API allows multiple fields to be queried
    at once. If any of the underlying fieldIds are populated by the same driver call, the results
    for those field IDs will be populated from a single call rather than making a driver call for
    each fieldId.

    Raises:
        NVMLError_InvalidArgument:
            If device or field_ids is invalid.
    """
    field_values = nvmlQuery('nvmlDeviceGetFieldValues', handle, field_ids)

    if not nvmlCheckReturn(field_values):
        timestamp = _time.time_ns() // 1000
        return [(NA, timestamp) for _ in range(len(field_ids))]

    values_with_timestamps: list[tuple[float | int | NaType, int]] = []
    for field_value in field_values:
        timestamp = field_value.timestamp
        if field_value.nvmlReturn != NVML_SUCCESS:
            value = NA
            timestamp = _time.time_ns() // 1000
        elif field_value.valueType == NVML_VALUE_TYPE_DOUBLE:
            value = field_value.value.dVal
        elif field_value.valueType == NVML_VALUE_TYPE_UNSIGNED_INT:
            value = field_value.value.uiVal
        elif field_value.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG:
            value = field_value.value.ulVal
        elif field_value.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
            value = field_value.value.ullVal
        elif field_value.valueType == NVML_VALUE_TYPE_SIGNED_LONG_LONG:
            value = field_value.value.llVal
        elif field_value.valueType == NVML_VALUE_TYPE_SIGNED_INT:
            value = field_value.value.iVal
        else:
            value = NA
        values_with_timestamps.append((value, timestamp))
    return values_with_timestamps


def nvmlCheckReturn(
    retval: _Any,
    types: type | tuple[type, ...] | None = None,
) -> bool:
    """Check whether the return value is not :const:`nvitop.NA` and is one of the given types."""
    if types is None:
        return retval != NA
    return retval != NA and isinstance(retval, types)


# Patch layers for backward compatibility ##########################################################
_pynvml_installation_corrupted: bool = not callable(
    getattr(_pynvml, '_nvmlGetFunctionPointer', None),
)

# Patch function `nvmlDeviceGet{Compute,Graphics,MPSCompute}RunningProcesses`
if not _pynvml_installation_corrupted:
    # pylint: disable-next=missing-class-docstring,too-few-public-methods,function-redefined
    class c_nvmlProcessInfo_v1_t(_pynvml._PrintableStructure):  # pylint: disable=protected-access
        _fields_: _ClassVar[list[tuple[str, type]]] = [
            # Process ID
            ('pid', _ctypes.c_uint),
            # Amount of used GPU memory in bytes.
            # Under WDDM, NVML_VALUE_NOT_AVAILABLE is always reported because Windows KMD manages
            # all the memory and not the NVIDIA driver.
            ('usedGpuMemory', _ctypes.c_ulonglong),
        ]
        _fmt_: _ClassVar[dict[str, str]] = {
            'usedGpuMemory': '%d B',
        }

    # pylint: disable-next=missing-class-docstring,too-few-public-methods,function-redefined
    class c_nvmlProcessInfo_v2_t(_pynvml._PrintableStructure):  # pylint: disable=protected-access
        _fields_: _ClassVar[list[tuple[str, type]]] = [
            # Process ID
            ('pid', _ctypes.c_uint),
            # Amount of used GPU memory in bytes.
            # Under WDDM, NVML_VALUE_NOT_AVAILABLE is always reported because Windows KMD manages
            # all the memory and not the NVIDIA driver.
            ('usedGpuMemory', _ctypes.c_ulonglong),
            # If MIG is enabled, stores a valid GPU instance ID. gpuInstanceId is set to 0xFFFFFFFF
            # otherwise.
            ('gpuInstanceId', _ctypes.c_uint),
            # If MIG is enabled, stores a valid compute instance ID. computeInstanceId is set to
            # 0xFFFFFFFF otherwise.
            ('computeInstanceId', _ctypes.c_uint),
        ]
        _fmt_: _ClassVar[dict[str, str]] = {
            'usedGpuMemory': '%d B',
        }

    # pylint: disable-next=missing-class-docstring,too-few-public-methods,function-redefined
    class c_nvmlProcessInfo_v3_t(_pynvml._PrintableStructure):  # pylint: disable=protected-access
        _fields_: _ClassVar[list[tuple[str, type]]] = [
            # Process ID
            ('pid', _ctypes.c_uint),
            # Amount of used GPU memory in bytes.
            # Under WDDM, NVML_VALUE_NOT_AVAILABLE is always reported because Windows KMD manages
            # all the memory and not the NVIDIA driver.
            ('usedGpuMemory', _ctypes.c_ulonglong),
            # If MIG is enabled, stores a valid GPU instance ID. gpuInstanceId is set to 0xFFFFFFFF
            # otherwise.
            ('gpuInstanceId', _ctypes.c_uint),
            # If MIG is enabled, stores a valid compute instance ID. computeInstanceId is set to
            # 0xFFFFFFFF otherwise.
            ('computeInstanceId', _ctypes.c_uint),
            # Amount of used GPU conf compute protected memory in bytes.
            ('usedGpuCcProtectedMemory', _ctypes.c_ulonglong),
        ]
        _fmt_: _ClassVar[dict[str, str]] = {
            'usedGpuMemory': '%d B',
            'usedGpuCcProtectedMemory': '%d B',
        }

    __get_running_processes_version_suffix = None
    c_nvmlProcessInfo_t = c_nvmlProcessInfo_v3_t

    def __determine_get_running_processes_version_suffix() -> str:
        global __get_running_processes_version_suffix, c_nvmlProcessInfo_t  # pylint: disable=global-statement

        if __get_running_processes_version_suffix is None:
            # pylint: disable-next=protected-access,no-member
            _nvmlGetFunctionPointer = _pynvml._nvmlGetFunctionPointer
            __get_running_processes_version_suffix = '_v3'

            def lookup(symbol: str) -> _Any | None:
                try:
                    ptr = _nvmlGetFunctionPointer(symbol)
                except NVMLError_FunctionNotFound:
                    LOGGER.debug('Failed to found symbol `%s`.', symbol)
                    return None
                LOGGER.debug('Found symbol `%s`.', symbol)
                return ptr

            if lookup('nvmlDeviceGetComputeRunningProcesses_v3'):
                if lookup('nvmlDeviceGetConfComputeMemSizeInfo') and not lookup(
                    'nvmlDeviceGetRunningProcessDetailList',
                ):
                    LOGGER.debug(
                        'NVML get running process version 3 API with v3 type struct is available.',
                    )
                else:
                    c_nvmlProcessInfo_t = c_nvmlProcessInfo_v2_t
                    LOGGER.debug(
                        'NVML get running process version 3 API with v3 type struct is not '
                        'available due to incompatible NVIDIA driver. Fallback to use get running '
                        'process version 3 API with v2 type struct.',
                    )
            else:
                c_nvmlProcessInfo_t = c_nvmlProcessInfo_v2_t
                __get_running_processes_version_suffix = '_v2'
                LOGGER.debug(
                    'NVML get running process version 3 API with v3 type struct is not available '
                    'due to incompatible NVIDIA driver. Fallback to use get running process '
                    'version 2 API with v2 type struct.',
                )
                if lookup('nvmlDeviceGetComputeRunningProcesses_v2'):
                    LOGGER.debug(
                        'NVML get running process version 2 API with v2 type struct is available.',
                    )
                else:
                    c_nvmlProcessInfo_t = c_nvmlProcessInfo_v1_t
                    __get_running_processes_version_suffix = ''
                    LOGGER.debug(
                        'NVML get running process version 2 API with v2 type struct is not '
                        'available due to incompatible NVIDIA driver. Fallback to use get '
                        'running process version 1 API with v1 type struct.',
                    )

        return __get_running_processes_version_suffix

    def __nvml_device_get_running_processes(
        func: str,
        handle: c_nvmlDevice_t,
    ) -> list[c_nvmlProcessInfo_t]:
        """Helper function for :func:`nvmlDeviceGet{Compute,Graphics,MPSCompute}RunningProcesses`.

        Modified from function :func:`pynvml.nvmlDeviceGetComputeRunningProcesses` in package
        `nvidia-ml-py <https://pypi.org/project/nvidia-ml-py>`_.
        """
        version_suffix = __determine_get_running_processes_version_suffix()

        # First call to get the size
        c_count = _ctypes.c_uint(0)
        # pylint: disable-next=protected-access
        fn = _pynvml._nvmlGetFunctionPointer(f'{func}{version_suffix}')
        ret = fn(handle, _ctypes.byref(c_count), None)

        if ret == NVML_SUCCESS:
            # Special case, no running processes
            return []
        if ret == NVML_ERROR_INSUFFICIENT_SIZE:
            # Typical case
            # Oversize the array in case more processes are created
            c_count.value = c_count.value * 2 + 5
            process_array = c_nvmlProcessInfo_t * c_count.value  # type: ignore[operator]
            c_processes = process_array()  # type: ignore[operator]

            # Make the call again
            ret = fn(handle, _ctypes.byref(c_count), c_processes)
            _pynvml._nvmlCheckReturn(ret)  # pylint: disable=protected-access

            processes = []
            for i in range(c_count.value):
                # Use an alternative struct for this object
                obj = _pynvml.nvmlStructToFriendlyObject(c_processes[i])
                if obj.usedGpuMemory == ULONGLONG_MAX:
                    # Special case for WDDM on Windows, see comment above
                    obj.usedGpuMemory = None
                processes.append(obj)

            return processes

        # Error case
        raise NVMLError(ret)

    def nvmlDeviceGetComputeRunningProcesses(  # pylint: disable=function-redefined
        handle: c_nvmlDevice_t,
    ) -> list[c_nvmlProcessInfo_t]:
        """Get information about processes with a compute context on a device.

        Note:
            - In MIG mode, if device handle is provided, the API returns aggregate information, only
              if the caller has appropriate privileges. Per-instance information can be queried by
              using specific MIG device handles.

        Raises:
            NVMLError_Uninitialized:
                If NVML was not first initialized with :func:`nvmlInit`.
            NVMLError_NoPermission:
                If the user doesn't have permission to perform this operation.
            NVMLError_InvalidArgument:
                If device is invalid.
            NVMLError_GpuIsLost:
                If the target GPU has fallen off the bus or is otherwise inaccessible.
            NVMLError_Unknown:
                On any unexpected error.
        """
        return __nvml_device_get_running_processes(
            'nvmlDeviceGetComputeRunningProcesses',
            handle,
        )

    def nvmlDeviceGetGraphicsRunningProcesses(  # pylint: disable=function-redefined
        handle: c_nvmlDevice_t,
    ) -> list[c_nvmlProcessInfo_t]:
        """Get information about processes with a graphics context on a device.

        Note:
            - In MIG mode, if device handle is provided, the API returns aggregate information, only
              if the caller has appropriate privileges. Per-instance information can be queried by
              using specific MIG device handles.

        Raises:
            NVMLError_Uninitialized:
                If NVML was not first initialized with :func:`nvmlInit`.
            NVMLError_NoPermission:
                If the user doesn't have permission to perform this operation.
            NVMLError_InvalidArgument:
                If device is invalid.
            NVMLError_GpuIsLost:
                If the target GPU has fallen off the bus or is otherwise inaccessible.
            NVMLError_Unknown:
                On any unexpected error.
        """
        return __nvml_device_get_running_processes(
            'nvmlDeviceGetGraphicsRunningProcesses',
            handle,
        )

    def nvmlDeviceGetMPSComputeRunningProcesses(  # pylint: disable=function-redefined
        handle: c_nvmlDevice_t,
    ) -> list[c_nvmlProcessInfo_t]:
        """Get information about processes with a MPS compute context on a device.

        Note:
            - In MIG mode, if device handle is provided, the API returns aggregate information, only
              if the caller has appropriate privileges. Per-instance information can be queried by
              using specific MIG device handles.

        Raises:
            NVMLError_Uninitialized:
                If NVML was not first initialized with :func:`nvmlInit`.
            NVMLError_NoPermission:
                If the user doesn't have permission to perform this operation.
            NVMLError_InvalidArgument:
                If device is invalid.
            NVMLError_GpuIsLost:
                If the target GPU has fallen off the bus or is otherwise inaccessible.
            NVMLError_Unknown:
                On any unexpected error.
        """
        return __nvml_device_get_running_processes(
            'nvmlDeviceGetMPSComputeRunningProcesses',
            handle,
        )

else:
    LOGGER.warning(
        'Your installed package `nvidia-ml-py` is corrupted. '
        'Skip patch functions `nvmlDeviceGet{Compute,Graphics,MPSCompute}RunningProcesses`. '
        'You may get incorrect or incomplete results. Please consider reinstall package '
        '`nvidia-ml-py` via `pip3 install --force-reinstall nvidia-ml-py nvitop`.',
    )

# Patch function `nvmlDeviceGetMemoryInfo`
if not _pynvml_installation_corrupted:
    # pylint: disable-next=missing-class-docstring,too-few-public-methods,function-redefined
    class c_nvmlMemory_v1_t(_pynvml._PrintableStructure):  # pylint: disable=protected-access
        _fields_: _ClassVar[list[tuple[str, type]]] = [
            # Total physical device memory (in bytes).
            ('total', _ctypes.c_ulonglong),
            # Unallocated device memory (in bytes).
            ('free', _ctypes.c_ulonglong),
            # Allocated device memory (in bytes).
            # Note that the driver/GPU always sets aside a small amount of memory for bookkeeping.
            ('used', _ctypes.c_ulonglong),
        ]
        _fmt_: _ClassVar[dict[str, str]] = {'<default>': '%d B'}

    # pylint: disable-next=missing-class-docstring,too-few-public-methods,function-redefined
    class c_nvmlMemory_v2_t(_pynvml._PrintableStructure):  # pylint: disable=protected-access
        _fields_: _ClassVar[list[tuple[str, type]]] = [
            # Structure format version (must be 2).
            ('version', _ctypes.c_uint),
            # Total physical device memory (in bytes).
            ('total', _ctypes.c_ulonglong),
            # Device memory (in bytes) reserved for system use (driver or firmware).
            ('reserved', _ctypes.c_ulonglong),
            # Unallocated device memory (in bytes).
            ('free', _ctypes.c_ulonglong),
            # Allocated device memory (in bytes).
            # Note that the driver/GPU always sets aside a small amount of memory for bookkeeping.
            ('used', _ctypes.c_ulonglong),
        ]
        _fmt_: _ClassVar[dict[str, str]] = {'<default>': '%d B'}

    nvmlMemory_v2 = getattr(_pynvml, 'nvmlMemory_v2', _ctypes.sizeof(c_nvmlMemory_v2_t) | 2 << 24)
    __get_memory_info_version_suffix = None
    c_nvmlMemory_t = c_nvmlMemory_v2_t

    def __determine_get_memory_info_version_suffix() -> str:
        global __get_memory_info_version_suffix, c_nvmlMemory_t  # pylint: disable=global-statement

        if __get_memory_info_version_suffix is None:
            # pylint: disable-next=protected-access,no-member
            _nvmlGetFunctionPointer = _pynvml._nvmlGetFunctionPointer
            __get_memory_info_version_suffix = '_v2'
            try:
                _nvmlGetFunctionPointer('nvmlDeviceGetMemoryInfo_v2')
            except NVMLError_FunctionNotFound:
                LOGGER.debug('Failed to found symbol `nvmlDeviceGetMemoryInfo_v2`.')
                c_nvmlMemory_t = c_nvmlMemory_v1_t
                __get_memory_info_version_suffix = ''
                LOGGER.debug(
                    'NVML get memory info version 2 API is not available due to incompatible '
                    'NVIDIA driver. Fallback to use NVML get memory info version 1 API.',
                )
            else:
                LOGGER.debug('Found symbol `nvmlDeviceGetMemoryInfo_v2`.')
                LOGGER.debug('NVML get memory info version 2 is available.')

        return __get_memory_info_version_suffix

    def nvmlDeviceGetMemoryInfo(  # pylint: disable=function-redefined
        handle: c_nvmlDevice_t,
    ) -> c_nvmlMemory_t:
        """Retrieve the amount of used, free, reserved and total memory available on the device, in bytes.

        Note:
            - The version 2 API adds additional memory information. The reserved amount is supported
              on version 2 only.
            - In MIG mode, if device handle is provided, the API returns aggregate information, only
              if the caller has appropriate privileges. Per-instance information can be queried by
              using specific MIG device handles.

        Raises:
            NVMLError_Uninitialized:
                If NVML was not first initialized with :func:`nvmlInit`.
            NVMLError_NoPermission:
                If the user doesn't have permission to perform this operation.
            NVMLError_InvalidArgument:
                If device is invalid.
            NVMLError_GpuIsLost:
                If the target GPU has fallen off the bus or is otherwise inaccessible.
            NVMLError_Unknown:
                On any unexpected error.
        """
        version_suffix = __determine_get_memory_info_version_suffix()
        if version_suffix == '_v2':
            c_memory = c_nvmlMemory_v2_t()
            c_memory.version = nvmlMemory_v2  # pylint: disable=attribute-defined-outside-init
            # pylint: disable-next=protected-access
            fn = _pynvml._nvmlGetFunctionPointer('nvmlDeviceGetMemoryInfo_v2')
        elif version_suffix in {'_v1', ''}:
            c_memory = c_nvmlMemory_v1_t()
            # pylint: disable-next=protected-access
            fn = _pynvml._nvmlGetFunctionPointer('nvmlDeviceGetMemoryInfo')
        else:
            raise ValueError(
                f'Unknown version suffix {version_suffix!r} for '
                'function `nvmlDeviceGetMemoryInfo`.',
            )
        ret = fn(handle, _ctypes.byref(c_memory))
        _pynvml._nvmlCheckReturn(ret)  # pylint: disable=protected-access
        return c_memory

else:
    LOGGER.warning(
        'Your installed package `nvidia-ml-py` is corrupted. '
        'Skip patch functions `nvmlDeviceGetMemoryInfo`. '
        'You may get incorrect or incomplete results. Please consider reinstall package '
        '`nvidia-ml-py` via `pip3 install --force-reinstall nvidia-ml-py nvitop`.',
    )


# Add support for lookup fallback and context manager ##############################################
class _CustomModule(_ModuleType):
    """Modified module type to support lookup fallback and context manager.

    Automatic lookup fallback:

        >>> libnvml.c_nvmlGpuInstance_t  # fallback to pynvml.c_nvmlGpuInstance_t
        <class 'pynvml.LP_struct_c_nvmlGpuInstance_t'>

    Context manager:

        >>> with libnvml:
        ...     handle = libnvml.nvmlDeviceGetHandleByIndex(0)
        ... # The NVML context has been shutdown
    """

    def __getattribute__(self, name: str) -> _Any | _Callable[..., _Any]:
        """Get a member from the current module. Fallback to the original package if missing."""
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(_pynvml, name)

    def __enter__(self) -> _Self:
        """Entry of the context manager for ``with`` statement."""
        _lazy_init()
        return self

    def __exit__(self, *exc: object) -> None:
        """Shutdown the NVML context in the context manager for ``with`` statement."""
        try:
            nvmlShutdown()
        except NVMLError:
            pass


# Replace entry in sys.modules for this module with an instance of _CustomModule
__modself = _sys.modules[__name__]
__modself.__class__ = _CustomModule
