# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import logging
import threading
from types import FunctionType
from typing import Tuple, Callable, Union, Optional, Any

import pynvml

from nvitop.core.utils import NA


__all__ = ['libnvml', 'nvml']


class libnvml(object):
    LOGGER = logging.getLogger('NVML')
    FLAGS = set()
    UNKNOWN_FUNCTIONS = set()
    NVMLError = pynvml.NVMLError
    NVMLError_LibraryNotFound = pynvml.NVMLError_LibraryNotFound  # pylint: disable=no-member

    def __new__(cls) -> 'libnvml':
        if not hasattr(cls, '_instance'):
            instance = cls._instance = super().__new__(cls)
            instance._initialized = False
            instance._lib_lock = threading.RLock()
            for name, attr in vars(pynvml).items():
                if name in ('nvmlInit', 'nvmlInitWithFlags', 'nvmlShutdown'):
                    continue
                if (
                    name.startswith('NVML_') or name.startswith('NVMLError_')
                ) or (
                    name.startswith('nvml') and isinstance(attr, FunctionType)
                ):
                    setattr(instance, name, attr)

        return cls._instance

    def __del__(self) -> None:
        try:
            self.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def __enter__(self) -> 'libnvml':
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.__del__()

    def __getattr__(self, name: str) -> Union[Any, Callable[..., Any]]:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(pynvml, name)

    def nvmlInit(self) -> None:
        self.nvmlInitWithFlags(0)

    def nvmlInitWithFlags(self, flags: int) -> None:
        with self._lib_lock:
            if self._initialized and flags in self.FLAGS:
                return

        try:
            pynvml.nvmlInitWithFlags(flags)
        except pynvml.NVMLError_LibraryNotFound:  # pylint: disable=no-member
            self.LOGGER.critical(
                'FATAL ERROR: NVIDIA Management Library (NVML) not found.\n'
                'HINT: The NVIDIA Management Library ships with the NVIDIA display driver (available at\n'
                '      https://www.nvidia.com/Download/index.aspx), or can be downloaded as part of the\n'
                '      NVIDIA CUDA Toolkit (available at https://developer.nvidia.com/cuda-downloads).\n'
                '      The lists of OS platforms and NVIDIA-GPUs supported by the NVML library can be\n'
                '      found in the NVML API Reference at https://docs.nvidia.com/deploy/nvml-api.'
            )
            raise
        else:
            with self._lib_lock:
                self._initialized = True  # pylint: disable=attribute-defined-outside-init
                self.FLAGS.add(flags)

    def nvmlShutdown(self) -> None:
        pynvml.nvmlShutdown()
        with self._lib_lock:
            self._initialized = False  # pylint: disable=attribute-defined-outside-init

    def nvmlQuery(self, func: Union[str, Callable[..., Any]], *args,
                  default: Any = NA, ignore_errors: bool = True, **kwargs) -> Any:
        if isinstance(func, str):
            func = getattr(pynvml, func)

        self.nvmlInit()

        try:
            retval = func(*args, **kwargs)
        except pynvml.NVMLError_FunctionNotFound:  # pylint: disable=no-member
            if func not in self.UNKNOWN_FUNCTIONS:
                self.UNKNOWN_FUNCTIONS.add(func)
                self.LOGGER.error(
                    'ERROR: A FunctionNotFound error occurred while calling %s.\n'
                    'Please verify whether the `nvidia-ml-py` package is compatible with your NVIDIA driver version.',
                    'nvmlQuery({!r}, *args, **kwargs)'.format(func)
                )
            if ignore_errors:
                return default
            raise
        except pynvml.NVMLError:
            if ignore_errors:
                return default
            raise
        else:
            if isinstance(retval, bytes):
                retval = retval.decode('UTF-8')
            return retval

    @staticmethod
    def nvmlCheckReturn(retval: Any, types: Optional[Union[type, Tuple[type, ...]]] = None) -> bool:
        if types is None:
            return retval != NA
        return retval != NA and isinstance(retval, types)


nvml = libnvml()
