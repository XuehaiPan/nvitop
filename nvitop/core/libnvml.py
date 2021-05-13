# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import sys

import pynvml


__all__ = ['libnvml', 'nvml']


class libnvml(object):
    def __init__(self):
        self._initialized = False

    def __del__(self):
        try:
            self.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(pynvml, name)

    def nvmlInit(self):
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
            self._initialized = True

    def nvmlShutdown(self):
        pynvml.nvmlShutdown()
        self._initialized = False

    def nvmlQuery(self, func, *args, catch_error=True, **kwargs):
        if isinstance(func, str):
            func = getattr(pynvml, func)

        self.nvmlInit()

        try:
            retval = func(*args, **kwargs)
        except pynvml.NVMLError:
            if catch_error:
                return 'N/A'
            raise
        else:
            if isinstance(retval, bytes):
                retval = retval.decode('UTF-8')
            return retval

    @staticmethod
    def nvmlCheckReturn(retval, types=None):
        if types is None:
            return retval != 'N/A'
        return retval != 'N/A' and isinstance(retval, types)


nvml = libnvml()
