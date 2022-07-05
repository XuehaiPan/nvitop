# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Utilities for the NVML Python bindings."""

# pylint: disable=invalid-name

import inspect
import logging
import re
import threading
from types import FunctionType
from typing import Tuple, Callable, Union, Optional, Any

# Python Bindings for the NVIDIA Management Library (NVML)
# https://pypi.org/project/nvidia-ml-py
import pynvml

from nvitop.core.utils import NA, colored


__all__ = ['libnvml', 'nvml', 'nvmlCheckReturn', 'NVMLError']


class libnvml:
    """The helper singleton class that holds members from package ``nvidia-ml-py``."""

    NVMLError = pynvml.NVMLError
    """Base exception class for NVML query errors."""

    LOGGER = logging.getLogger('NVML')
    UNKNOWN_FUNCTIONS = {}
    UNKNOWN_FUNCTIONS_CACHE_SIZE = 1024
    VERSIONED_PATTERN = re.compile(r'^(?P<name>\w+)(?P<suffix>_v(\d)+)$')

    c_nvmlDevice_t = pynvml.c_nvmlDevice_t

    def __new__(cls) -> 'libnvml':
        """Gets the singleton instance of ``libnvml``."""

        if not hasattr(cls, '_instance'):
            instance = cls._instance = super().__new__(cls)
            instance._flags = []
            instance._initialized = False
            instance._lock = threading.Lock()
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
        """Automatically shutdowns the NVML context on destruction."""

        try:
            self.nvmlShutdown()
        except nvml.NVMLError:
            pass

    def __enter__(self) -> 'libnvml':
        """Entry of the context manager for ``with`` statement."""

        self._lazy_init()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Shutdowns the NVML context in the context manager for ``with`` statement."""

        self.__del__()

    def __getattr__(self, name: str) -> Union[Any, Callable[..., Any]]:
        """Gets a member from the instance. Fallback to the original package if missing."""

        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(pynvml, name)

    def _lazy_init(self) -> None:
        """Lazily initializes the NVML context."""

        with self._lock:
            if self._initialized:
                return
        self.nvmlInit()

    def nvmlInit(self) -> None:
        """Initializes the NVML context with default flag (0).

        Raises:
            NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            AttributeError:
                If cannot find function ``nvmlInitWithFlags``, usually the ``pynvml`` module is
                overridden by other modules. Need to reinstall package ``nvidia-ml-py``.
        """

        self.nvmlInitWithFlags(0)

    def nvmlInitWithFlags(self, flags: int) -> None:
        """Initializes the NVML context with the given flags.

        Raises:
            NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            AttributeError:
                If cannot find function ``nvmlInitWithFlags``, usually the ``pynvml`` module is
                overridden by other modules. Need to reinstall package ``nvidia-ml-py``.
        """

        with self._lock:
            if len(self._flags) > 0 and flags == self._flags[-1]:
                self._initialized = True  # pylint: disable=attribute-defined-outside-init
                return

        try:
            pynvml.nvmlInitWithFlags(flags)
        except nvml.NVMLError_LibraryNotFound:  # pylint: disable=no-member
            message = '\n'.join((
                'FATAL ERROR: NVIDIA Management Library (NVML) not found.',
                'HINT: The NVIDIA Management Library ships with the NVIDIA display driver (available at',
                '      https://www.nvidia.com/Download/index.aspx), or can be downloaded as part of the',
                '      NVIDIA CUDA Toolkit (available at https://developer.nvidia.com/cuda-downloads).',
                '      The lists of OS platforms and NVIDIA-GPUs supported by the NVML library can be',
                '      found in the NVML API Reference at https://docs.nvidia.com/deploy/nvml-api.',
            ))
            for text, color, attrs in (('FATAL ERROR:', 'red', ('bold',)),
                                       ('HINT:', 'yellow', ('bold',)),
                                       ('https://www.nvidia.com/Download/index.aspx', None, ('underline',)),
                                       ('https://developer.nvidia.com/cuda-downloads', None, ('underline',)),
                                       ('https://docs.nvidia.com/deploy/nvml-api', None, ('underline',))):
                message = message.replace(text, colored(text, color=color, attrs=attrs))

            self.LOGGER.critical(message)
            raise
        except AttributeError:
            message = '\n'.join((
                'FATAL ERROR: The dependency package `nvidia-ml-py` is corrupted. You may have installed',
                '             other packages overriding the module `pynvml`.',
                'Please reinstall `nvitop` with command:',
                '    python3 -m pip install --force-reinstall nvitop',
            ))
            for text, color, attrs in (('FATAL ERROR:', 'red', ('bold',)),
                                       ('nvidia-ml-py', None, ('bold',)),
                                       ('pynvml', None, ('bold',)),
                                       ('nvitop', None, ('bold',))):
                message = message.replace(text, colored(text, color=color, attrs=attrs), 1)

            self.LOGGER.critical(message)
            raise
        else:
            with self._lock:
                self._flags.append(flags)
                self._initialized = True  # pylint: disable=attribute-defined-outside-init

    def nvmlShutdown(self) -> None:
        """Shutdowns the NVML context.

        Raises:
            NVMLError_LibraryNotFound:
                If cannot find the NVML library, usually the NVIDIA driver is not installed.
            NVMLError_DriverNotLoaded:
                If NVIDIA driver is not loaded.
            NVMLError_LibRmVersionMismatch:
                If RM detects a driver/library version mismatch, usually after an upgrade for NVIDIA
                driver without reloading the kernel module.
            NVMLError_Uninitialized:
                If NVML was not first initialized with ``nvmlInit()``.
        """

        pynvml.nvmlShutdown()
        with self._lock:
            try:
                self._flags.pop()
            except IndexError:
                pass
            self._initialized = (len(self._flags) > 0)  # pylint: disable=attribute-defined-outside-init

    def nvmlQuery(self, func: Union[Callable[..., Any], str],
                  *args,
                  default: Any = NA,
                  ignore_errors: bool = True,
                  ignore_function_not_found: bool = False,
                  **kwargs) -> Any:
        """Calls a function with the given arguments from NVML. The NVML context will be lazily initialized.

        Args:
            func (Union[Callable[..., Any], str]):
                The function to call. If it is given by string, lookup for the
                function first from ``pynvml``.
            default (Any):
                The default value if the query fails.
            ignore_errors (bool):
                Whether to ignore errors and return the default value.
            ignore_function_not_found (bool):
                Whether to ignore function not found errors and return the
                default value. If set to ``False``, a error message will be logged
                to the logger.
            *args:
                Positional arguments to pass to the query function.
            **kwargs:
                Keyword arguments to pass to the query function.
        """

        self._lazy_init()

        try:
            if isinstance(func, str):
                try:
                    func = getattr(self, func)
                except AttributeError as e1:
                    raise nvml.NVMLError_FunctionNotFound from e1  # pylint: disable=no-member

            retval = func(*args, **kwargs)
        except nvml.NVMLError_FunctionNotFound as e2:  # pylint: disable=no-member
            if not ignore_function_not_found:
                if identifier.__name__ == '<lambda>':
                    identifier = inspect.getsource(func)
                else:
                    identifier = repr(func)
                with self._lock:
                    if (
                        identifier not in self.UNKNOWN_FUNCTIONS
                        and len(self.UNKNOWN_FUNCTIONS) < self.UNKNOWN_FUNCTIONS_CACHE_SIZE
                    ):
                        self.UNKNOWN_FUNCTIONS[identifier] = (func, e2)
                        self.LOGGER.error(
                            'ERROR: A FunctionNotFound error occurred while calling %s.\n'
                            'Please verify whether the `nvidia-ml-py` package is '
                            'compatible with your NVIDIA driver version.',
                            'nvmlQuery({!r}, *args, **kwargs)'.format(func)
                        )
            if ignore_errors or ignore_function_not_found:
                return default
            raise
        except nvml.NVMLError:
            if ignore_errors:
                return default
            raise
        else:
            if isinstance(retval, bytes):
                retval = retval.decode('UTF-8')
            return retval

    @staticmethod
    def nvmlCheckReturn(retval: Any, types: Optional[Union[type, Tuple[type, ...]]] = None) -> bool:
        """Checks the return value is not ``nvitop.NA`` and is one of the given types."""

        if types is None:
            return retval != NA
        return retval != NA and isinstance(retval, types)


nvml = libnvml()
"""The singleton instance of class ``libnvml``."""

nvmlCheckReturn = nvml.nvmlCheckReturn

NVMLError = nvml.NVMLError
