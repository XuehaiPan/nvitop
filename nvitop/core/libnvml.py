# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import sys
import threading
from typing import Tuple, Callable, Union, Optional, Any

import pynvml

from .utils import NA


__all__ = ['libnvml', 'nvml']


class libnvml(object):
    def __new__(cls) -> 'libnvml':
        if not hasattr(cls, '_instance'):
            instance = cls._instance = super().__new__(cls)
            instance._initialized = False
            instance._lib_lock = threading.RLock()
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
        with self._lib_lock:
            if self._initialized:
                return

        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError_LibraryNotFound:  # pylint: disable=no-member
            print('ERROR: NVIDIA Management Library (NVML) not found.\n'
                  'HINT: The NVIDIA Management Library ships with the NVIDIA display driver (available at\n'
                  '      https://www.nvidia.com/Download/index.aspx), or can be downloaded as part of the\n'
                  '      NVIDIA CUDA Toolkit (available at https://developer.nvidia.com/cuda-downloads).\n'
                  '      The lists of OS platforms and NVIDIA-GPUs supported by the NVML library can be\n'
                  '      found in the NVML API Reference at https://docs.nvidia.com/deploy/nvml-api.',
                  file=sys.stderr)
            raise
        else:
            with self._lib_lock:
                self._initialized = True  # pylint: disable=attribute-defined-outside-init

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
            print('ERROR: Function Not Found.\n'
                  'Please verify whether the `nvidia-ml-py` package is compatible with your NVIDIA driver version.',
                  file=sys.stderr)
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
