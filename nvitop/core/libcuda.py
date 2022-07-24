# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Python bindings for the `CUDA driver APIs <https://docs.nvidia.com/cuda/cuda-driver-api>`_."""

# pylint: disable=invalid-name

import ctypes as _ctypes
import platform as _platform
import string as _string
import sys as _sys
import threading as _threading
from typing import Any as _Any
from typing import Tuple as _Tuple
from typing import Type as _Type


# pylint: disable-next=missing-class-docstring,too-few-public-methods
class struct_c_CUdevice_t(_ctypes.Structure):
    pass  # opaque handle


c_CUdevice_t = _ctypes.POINTER(struct_c_CUdevice_t)

_CUresult_t = _ctypes.c_uint

## Error codes ##
# pylint: disable=line-too-long
CUDA_SUCCESS = 0
"""The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete (see :func:`cuEventQuery` and :func:`cuStreamQuery`)."""
CUDA_ERROR_INVALID_VALUE = 1
"""This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values."""
CUDA_ERROR_OUT_OF_MEMORY = 2
"""The API call failed because it was unable to allocate enough memory to perform the requested operation."""
CUDA_ERROR_NOT_INITIALIZED = 3
"""This indicates that the CUDA driver has not been initialized with :func:`cuInit` or that initialization has failed."""
CUDA_ERROR_DEINITIALIZED = 4
"""This indicates that the CUDA driver is in the process of shutting down."""
CUDA_ERROR_PROFILER_DISABLED = 5
"""This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler."""
CUDA_ERROR_STUB_LIBRARY = 34
"""This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error."""
CUDA_ERROR_DEVICE_UNAVAILABLE = 46
"""This indicates that requested CUDA device is unavailable at the current time. Devices are often unavailable due to use of :data:`CU_COMPUTEMODE_EXCLUSIVE_PROCESS` or :data:`CU_COMPUTEMODE_PROHIBITED`."""
CUDA_ERROR_NO_DEVICE = 100
"""This indicates that no CUDA - capable devices were detected by the installed CUDA driver."""
CUDA_ERROR_INVALID_DEVICE = 101
"""This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device."""
CUDA_ERROR_DEVICE_NOT_LICENSED = 102
"""This error indicates that the Grid license is not applied."""
CUDA_ERROR_INVALID_IMAGE = 200
"""This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module."""
CUDA_ERROR_INVALID_CONTEXT = 201
"""This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had :func:`cuCtxDestroy` invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See :func:`cuCtxGetApiVersion` for more details."""
CUDA_ERROR_MAP_FAILED = 205
"""This indicates that a map or register operation has failed."""
CUDA_ERROR_UNMAP_FAILED = 206
"""This indicates that an unmap or unregister operation has failed."""
CUDA_ERROR_ARRAY_IS_MAPPED = 207
"""This indicates that the specified array is currently mapped and thus cannot be destroyed."""
CUDA_ERROR_ALREADY_MAPPED = 208
"""This indicates that the resource is already mapped."""
CUDA_ERROR_NO_BINARY_FOR_GPU = 209
"""This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration."""
CUDA_ERROR_ALREADY_ACQUIRED = 210
"""This indicates that a resource has already been acquired."""
CUDA_ERROR_NOT_MAPPED = 211
"""This indicates that a resource is not mapped."""
CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
"""This indicates that a mapped resource is not available for access as an array."""
CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
"""This indicates that a mapped resource is not available for access as a pointer."""
CUDA_ERROR_ECC_UNCORRECTABLE = 214
"""This indicates that an uncorrectable ECC error was detected during execution."""
CUDA_ERROR_UNSUPPORTED_LIMIT = 215
"""This indicates that the :class:`CUlimit` passed to the API call is not supported by the active device."""
CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
"""This indicates that the :class:`CUcontext` passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread."""
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
"""This indicates that peer access is not supported across the given devices."""
CUDA_ERROR_INVALID_PTX = 218
"""This indicates that a PTX JIT compilation failed."""
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
"""This indicates an error with OpenGL or DirectX context."""
CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
"""This indicates that an uncorrectable NVLink error was detected during the execution."""
CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
"""This indicates that the PTX JIT compiler library was not found."""
CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
"""This indicates that the provided PTX was compiled with an unsupported toolchain."""
CUDA_ERROR_JIT_COMPILATION_DISABLED = 223
"""This indicates that the PTX JIT compilation was disabled."""
CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224
"""This indicates that the :class:`CUexecAffinityType` passed to the API call is not supported by the active device."""
CUDA_ERROR_INVALID_SOURCE = 300
"""This indicates that the device kernel source is invalid. This includes compilation / linker errors encountered in device code or user error."""
CUDA_ERROR_FILE_NOT_FOUND = 301
"""This indicates that the file specified was not found."""
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
"""This indicates that a link to a shared object failed to resolve."""
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
"""This indicates that initialization of a shared object failed."""
CUDA_ERROR_OPERATING_SYSTEM = 304
"""This indicates that an OS call failed."""
CUDA_ERROR_INVALID_HANDLE = 400
"""This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like :class:`CUstream` and :class:`CUevent`."""
CUDA_ERROR_ILLEGAL_STATE = 401
"""This indicates that a resource required by the API call is not in a valid state to perform the requested operation."""
CUDA_ERROR_NOT_FOUND = 500
"""This indicates that a named symbol was not found. Examples of symbols are global / constant variable names, driver function names, texture names, and surface names."""
CUDA_ERROR_NOT_READY = 600
"""This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than :data:`CUDA_SUCCESS` (which indicates completion). Calls that may return this value include :func:`cuEventQuery` and :func:`cuStreamQuery`."""
CUDA_ERROR_ILLEGAL_ADDRESS = 700
"""While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
"""This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64 - bit pointer when a 32 - bit int is expected) is equivalent to passing too many arguments and can also result in this error."""
CUDA_ERROR_LAUNCH_TIMEOUT = 702
"""This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute :data:`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT` for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
"""This error indicates a kernel launch that uses an incompatible texturing mode."""
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
"""This error indicates that a call to :func:`cuCtxEnablePeerAccess` is trying to re - enable peer access to a context which has already had peer access to it enabled."""
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
"""This error indicates that :func:`cuCtxDisablePeerAccess` is trying to disable peer access which has not been enabled yet via :func:`cuCtxEnablePeerAccess`."""
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
"""This error indicates that the primary context for the specified device has already been initialized."""
CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
"""This error indicates that the context current to the calling thread has been destroyed using :func:`cuCtxDestroy`, or is a primary context which has not yet been initialized."""
CUDA_ERROR_ASSERT = 710
"""A device - side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA."""
CUDA_ERROR_TOO_MANY_PEERS = 711
"""This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to :func:`cuCtxEnablePeerAccess`."""
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
"""This error indicates that the memory range passed to :func:`cuMemHostRegister` has already been registered."""
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
"""This error indicates that the pointer passed to :func:`cuMemHostUnregister` does not correspond to any currently registered memory region."""
CUDA_ERROR_HARDWARE_STACK_ERROR = 714
"""While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
"""While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_MISALIGNED_ADDRESS = 716
"""While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
"""While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_INVALID_PC = 718
"""While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_LAUNCH_FAILED = 719
"""An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
"""This error indicates that the number of blocks launched per grid for a kernel that was launched via either :func:`cuLaunchCooperativeKernel` or :func:`cuLaunchCooperativeKernelMultiDevice` exceeds the maximum number of blocks as allowed by :func:`cuOccupancyMaxActiveBlocksPerMultiprocessor` or :func:`cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` times the number of multiprocessors as specified by the device attribute :data:`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`."""
CUDA_ERROR_NOT_PERMITTED = 800
"""This error indicates that the attempted operation is not permitted."""
CUDA_ERROR_NOT_SUPPORTED = 801
"""This error indicates that the attempted operation is not supported on the current system or device."""
CUDA_ERROR_SYSTEM_NOT_READY = 802
"""This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide."""
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
"""This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions."""
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
"""This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the ``CUDA_VISIBLE_DEVICES`` environment variable."""
CUDA_ERROR_MPS_CONNECTION_FAILED = 805
"""This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server."""
CUDA_ERROR_MPS_RPC_FAILURE = 806
"""This error indicates that the remote procedural call between the MPS server and the MPS client failed."""
CUDA_ERROR_MPS_SERVER_NOT_READY = 807
"""This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure."""
CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808
"""This error indicates that the hardware resources required to create MPS client have been exhausted."""
CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809
"""This error indicates the the hardware resources required to support device connections have been exhausted."""
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
"""This error indicates that the operation is not permitted when the stream is capturing."""
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
"""This error indicates that the current capture sequence on the stream has been invalidated due to a previous error."""
CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
"""This error indicates that the operation would have resulted in a merge of two independent capture sequences."""
CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
"""This error indicates that the capture was initiated not in this stream."""
CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
"""This error indicates that the capture sequence contains a fork that was not joined to the primary stream."""
CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
"""This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in -stream ordering dependencies are allowed to cross the boundary."""
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
"""This error indicates a disallowed implicit dependency on a current capture sequence from :func:`cudaStreamLegacy`."""
CUDA_ERROR_CAPTURED_EVENT = 907
"""This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream."""
CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
"""A stream capture sequence not initiated with the :data:`CU_STREAM_CAPTURE_MODE_RELAXED` argument to :func:`cuStreamBeginCapture` was passed to :func:`cuStreamEndCapture` in a different thread."""
CUDA_ERROR_TIMEOUT = 909
"""This error indicates that the timeout specified for the wait operation has lapsed."""
CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
"""This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update."""
CUDA_ERROR_EXTERNAL_DEVICE = 911
"""This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
CUDA_ERROR_UNKNOWN = 999
"""This indicates that an unknown internal error has occurred."""
# pylint: enable=line-too-long


## Error Checking ##
class CUDAError(Exception):
    """Base exception class for CUDA driver query errors."""

    _value_class_mapping = {}
    _errcode_to_string = {  # List of currently known error codes
        CUDA_ERROR_NOT_INITIALIZED:                'Initialization error.',
        CUDA_ERROR_NOT_FOUND:                      'Named symbol not found.',
        CUDA_ERROR_INVALID_VALUE:                  'Invalid argument.',
        CUDA_ERROR_NO_DEVICE:                      'No CUDA-capable device is detected.',
        CUDA_ERROR_INVALID_DEVICE:                 'Invalid device ordinal.',
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:         'System has unsupported display driver / CUDA driver combination.',
        CUDA_ERROR_DEINITIALIZED:                  'Driver shutting down.',
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: 'Forward compatibility was attempted on non supported Hardware.',
        CUDA_ERROR_INVALID_CONTEXT:                'Invalid device context.',
    }  # fmt:skip
    _errcode_to_name = {}

    def __new__(cls, value: int) -> 'CUDAError':
        """Maps value to a proper subclass of :class:`CUDAError`."""

        if cls is CUDAError:
            # pylint: disable-next=self-cls-assignment
            cls = CUDAError._value_class_mapping.get(value, cls)
        obj = Exception.__new__(cls)
        obj.value = value
        return obj

    def __str__(self) -> str:
        # pylint: disable=no-member
        try:
            if self.value not in CUDAError._errcode_to_string:
                CUDAError._errcode_to_string[self.value] = '{}.'.format(
                    cuGetErrorString(self.value).rstrip('.').capitalize()
                )
            if self.value not in CUDAError._errcode_to_name:
                CUDAError._errcode_to_name[self.value] = cuGetErrorName(self.value)
            return '{} Code: {} ({}).'.format(
                CUDAError._errcode_to_string[self.value],
                CUDAError._errcode_to_name[self.value],
                self.value,
            )
        except CUDAError:
            return 'CUDA Error with code {}.'.format(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CUDAError):
            return NotImplemented
        return self.value == other.value  # pylint: disable=no-member

    def __reduce__(self) -> _Tuple[_Type['CUDAError'], _Tuple[int]]:
        return CUDAError, (self.value,)  # pylint: disable=no-member


def cudaExceptionClass(cudaErrorCode: int) -> _Type[CUDAError]:
    """Maps value to a proper subclass of :class:`CUDAError`.

    Raises:
        ValueError: If the error code is not valid.
    """

    # pylint: disable=protected-access
    if cudaErrorCode not in CUDAError._value_class_mapping:
        raise ValueError('cudaErrorCode {} is not valid.'.format(cudaErrorCode))
    return CUDAError._value_class_mapping[cudaErrorCode]


def _extract_cuda_errors_as_classes() -> None:
    """Generates a hierarchy of classes on top of :class:`CUDAError` class.

    Each CUDA Error gets a new :class:`CUDAError` subclass. This way try-except blocks can filter
    appropriate exceptions more easily.

    :class:`CUDAError` is a parent class. Each ``CUDA_ERROR_*`` gets it's own subclass.
    e.g. :data:`CUDA_ERROR_INVALID_VALUE` will be turned into :class:`CUDAError_InvalidValue`.
    """

    this_module = _sys.modules[__name__]
    cuda_error_names = [x for x in dir(this_module) if x.startswith('CUDA_ERROR_')]
    for err_name in cuda_error_names:
        # e.g. Turn CUDA_ERROR_INVALID_VALUE into CUDAError_InvalidValue
        pascal_case = _string.capwords(err_name.replace('CUDA_ERROR_', ''), '_').replace('_', '')
        class_name = 'CUDAError_{}'.format(pascal_case)
        err_val = getattr(this_module, err_name)

        def gen_new(value):
            def new(cls):
                obj = CUDAError.__new__(cls, value)
                return obj

            return new

        # pylint: disable=protected-access
        new_error_class = type(class_name, (CUDAError,), {'__new__': gen_new(err_val)})
        new_error_class.__module__ = __name__
        if err_val in CUDAError._errcode_to_string:
            new_error_class.__doc__ = 'CUDA Error: {} Code: :data:`{}` ({}).'.format(
                CUDAError._errcode_to_string[err_val],
                err_name,
                err_val,
            )
        else:
            new_error_class.__doc__ = 'CUDA Error with code :data:`{}` ({})'.format(
                err_name, err_val
            )
        setattr(this_module, class_name, new_error_class)
        CUDAError._value_class_mapping[err_val] = new_error_class
        CUDAError._errcode_to_name[err_val] = err_name


_extract_cuda_errors_as_classes()
del _extract_cuda_errors_as_classes


def _cudaCheckReturn(ret: _Any) -> _Any:
    if ret != CUDA_SUCCESS:
        raise CUDAError(ret)
    return ret


## Function access ##
__cudaLib = None
__initialized = False
__libLoadLock = _threading.Lock()
# Function pointers are cached to prevent unnecessary libLoadLock locking
__cudaGetFunctionPointer_cache = {}


def __cudaGetFunctionPointer(name: str) -> _ctypes._CFuncPtr:
    """
    Get the function pointer from the CUDA driver library.

    Raises:
        CUDAError_NotInitialized:
            If cannot found the CUDA driver library.
        CUDAError_NotFound:
            If cannot found the function pointer.
    """

    if name in __cudaGetFunctionPointer_cache:
        return __cudaGetFunctionPointer_cache[name]

    with __libLoadLock:
        # Ensure library was loaded
        if __cudaLib is None:
            raise CUDAError(CUDA_ERROR_NOT_INITIALIZED)
        try:
            __cudaGetFunctionPointer_cache[name] = getattr(__cudaLib, name)
            return __cudaGetFunctionPointer_cache[name]
        except AttributeError as ex:
            raise CUDAError(CUDA_ERROR_NOT_FOUND) from ex


def __LoadCudaLibrary() -> None:
    """
    Load the library if it isn't loaded already.

    Raises:
        CUDAError_NotInitialized:
            If cannot found the CUDA driver library.
    """

    global __cudaLib  # pylint: disable=global-statement

    if __cudaLib is None:
        # Lock to ensure only one caller loads the library
        with __libLoadLock:
            # Ensure the library still isn't loaded
            if __cudaLib is None:
                # Platform specific libcuda location
                system = _platform.system()
                if system == 'Darwin':
                    lib_filenames = [
                        'libcuda.dylib',  # check library path first
                        '/usr/local/cuda/lib/libcuda.dylib',
                    ]
                elif system == 'Linux':
                    lib_filenames = [
                        'libcuda.so',  # check library path first
                        '/usr/lib64/nvidia/libcuda.so',  # Redhat/CentOS/Fedora
                        '/usr/lib/x86_64-linux-gnu/libcuda.so',  # Ubuntu
                        '/usr/lib/wsl/lib/libcuda.so',  # WSL
                    ]
                elif system == 'Windows':
                    lib_filenames = ['nvcuda.dll']
                # Open library
                for lib_filename in lib_filenames:
                    try:
                        __cudaLib = _ctypes.CDLL(lib_filename)
                        break
                    except OSError:
                        pass
                if __cudaLib is None:
                    _cudaCheckReturn(CUDA_ERROR_NOT_INITIALIZED)


def cuInit(flags: int = 0) -> None:
    """Initialize the CUDA driver API.

    Initializes the driver API and must be called before any other function from the driver API.
    Currently, the ``flags`` parameter must be :data:`0`. If :func:`cuInit` has not been called,
    any function from the driver API will return :data:`CUDA_ERROR_NOT_INITIALIZED`.

    Raises:
        CUDAError_NoDevice:
            If no CUDA-capable device is available.
        CUDAError_InvalidDevice:
            If the device ordinal supplied by the user does not correspond to a valid CUDA device or
            that the action requested is invalid for the specified device.
        CUDAError_SystemDriverMismatch:
            If there is a mismatch between the versions of the display driver and the CUDA driver.
        CUDAError_CompatNotSupportedOnDevice:
            If the system was upgraded to run with forward compatibility but the visible hardware
            detected by CUDA does not support this configuration.
        CUDAError_InvalidValue:
            If passed with invalid flag value.
        CUDAError_NotInitialized:
            If cannot found the CUDA driver library.
    """

    global __initialized  # pylint: disable=global-statement

    __LoadCudaLibrary()

    if __initialized:
        return

    fn = __cudaGetFunctionPointer('cuInit')

    ret = fn(_ctypes.c_uint(flags))
    _cudaCheckReturn(ret)

    with __libLoadLock:
        __initialized = True


def cuGetErrorName(error: int) -> str:
    """Gets the string representation of an error code enum name.

    Raises:
        CUDAError_InvalidValue:
            If the error code is not recognized.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    fn = __cudaGetFunctionPointer('cuGetErrorName')

    p_name = _ctypes.POINTER(_ctypes.c_char_p)()
    ret = fn(_CUresult_t(error), _ctypes.byref(p_name))
    _cudaCheckReturn(ret)
    name = _ctypes.string_at(p_name)
    return name.decode('UTF-8', errors='replace')


def cuGetErrorString(error: int) -> str:
    """Gets the string description of an error code.

    Raises:
        CUDAError_InvalidValue:
            If the error code is not recognized.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    fn = __cudaGetFunctionPointer('cuGetErrorString')

    p_name = _ctypes.POINTER(_ctypes.c_char_p)()
    ret = fn(_CUresult_t(error), _ctypes.byref(p_name))
    _cudaCheckReturn(ret)
    name = _ctypes.string_at(p_name)
    return name.decode('UTF-8', errors='replace')


def cuDriverGetVersion() -> str:
    """Returns the latest CUDA version supported by driver.

    Returns:
        A string of the form :data:`'<major>.<minor>'`.

    Raises:
        CUDAError_InvalidValue:
            If the driver call fails.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    fn = __cudaGetFunctionPointer('cuDriverGetVersion')

    driver_version = _ctypes.c_int()
    ret = fn(_ctypes.byref(driver_version))
    _cudaCheckReturn(ret)
    major = driver_version.value // 1000
    minor = (driver_version.value % 1000) // 10
    return '{}.{}'.format(major, minor)


def cuDeviceGetCount() -> int:
    """Returns the number of compute-capable devices.

    Returns: int
        The number of devices with compute capability greater than or equal to 2.0 that are available
        for execution. If there is no such device, :func:`cuDeviceGetCount` returns :data:`0`.

    Raises:
        CUDAError_InvalidContext:
            If there is no context bound to the current thread.
        CUDAError_InvalidValue:
            If the driver call fails.
        CUDAError_Deinitialized:
            If the CUDA driver in the process is shutting down.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    fn = __cudaGetFunctionPointer('cuDeviceGetCount')

    count = _ctypes.c_int(0)
    ret = fn(_ctypes.byref(count))
    _cudaCheckReturn(ret)
    return count.value


def cuDeviceGet(ordinal: int) -> c_CUdevice_t:
    """Returns a handle to a compute device.

    Returns a device handle given an ordinal in the range :code:`[0, ..., cuDeviceGetCount() - 1]`.

    Raises:
        CUDAError_InvalidContext:
            If there is no context bound to the current thread.
        CUDAError_InvalidDevice:
            If the device ordinal supplied by the user does not correspond to a valid CUDA device or
            that the action requested is invalid for the specified device.
        CUDAError_InvalidValue:
            If the driver call fails.
        CUDAError_Deinitialized:
            If the CUDA driver in the process is shutting down.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    fn = __cudaGetFunctionPointer('cuDeviceGet')

    device = c_CUdevice_t()
    ret = fn(_ctypes.byref(device), _ctypes.c_int(ordinal))
    _cudaCheckReturn(ret)
    return device


def cuDeviceGetName(device: c_CUdevice_t) -> str:
    """Returns an identifier string for the device.

    Returns an ASCII string identifying the device.

    Raises:
        CUDAError_InvalidContext:
            If there is no context bound to the current thread.
        CUDAError_InvalidDevice:
            If the device ordinal supplied by the user does not correspond to a valid CUDA device or
            that the action requested is invalid for the specified device.
        CUDAError_InvalidValue:
            If the driver call fails.
        CUDAError_Deinitialized:
            If the CUDA driver in the process is shutting down.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    fn = __cudaGetFunctionPointer('cuDeviceGetName')

    name = _ctypes.create_string_buffer(256)
    ret = fn(name, _ctypes.c_int(256), device)
    _cudaCheckReturn(ret)
    return name.value.decode('UTF-8', errors='replace')


def cuDeviceGetUuid(device: c_CUdevice_t) -> str:
    """Returns a UUID for the device.

    Raises:
        CUDAError_InvalidDevice:
            If the device ordinal supplied by the user does not correspond to a valid CUDA device or
            that the action requested is invalid for the specified device.
        CUDAError_InvalidValue:
            If the driver call fails.
        CUDAError_Deinitialized:
            If the CUDA driver in the process is shutting down.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    try:
        fn = __cudaGetFunctionPointer('cuDeviceGetUuid_v2')
    except AttributeError:
        fn = __cudaGetFunctionPointer('cuDeviceGetUuid')

    ubyte_array = _ctypes.c_ubyte * 16
    uuid = ubyte_array()
    ret = fn(uuid, device)
    _cudaCheckReturn(ret)
    uuid = ''.join(map('{:02x}'.format, uuid))
    return '-'.join((uuid[:8], uuid[8:12], uuid[12:16], uuid[16:20], uuid[20:32]))


def cuDeviceGetUuid_v2(device: c_CUdevice_t) -> str:
    """Returns a UUID for the device (CUDA 11.4+).

    Raises:
        CUDAError_InvalidDevice:
            If the device ordinal supplied by the user does not correspond to a valid CUDA device or
            that the action requested is invalid for the specified device.
        CUDAError_InvalidValue:
            If the driver call fails.
        CUDAError_Deinitialized:
            If the CUDA driver in the process is shutting down.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    fn = __cudaGetFunctionPointer('cuDeviceGetUuid_v2')

    ubyte_array = _ctypes.c_ubyte * 16
    uuid = ubyte_array()
    ret = fn(uuid, device)
    _cudaCheckReturn(ret)
    uuid = ''.join(map('{:0x}'.format, uuid.value))
    return '-'.join((uuid[:8], uuid[8:12], uuid[12:16], uuid[16:20], uuid[20:32]))


def cuDeviceTotalMem(device: c_CUdevice_t) -> int:
    """Returns the total amount of memory on the device (in bytes).

    Raises:
        CUDAError_InvalidContext:
            If there is no context bound to the current thread.
        CUDAError_InvalidDevice:
            If the device ordinal supplied by the user does not correspond to a valid CUDA device or
            that the action requested is invalid for the specified device.
        CUDAError_InvalidValue:
            If the driver call fails.
        CUDAError_Deinitialized:
            If the CUDA driver in the process is shutting down.
        CUDAError_NotInitialized:
            If the CUDA driver API is not initialized.
    """

    fn = __cudaGetFunctionPointer('cuDeviceTotalMem')

    bytes = _ctypes.c_size_t()  # pylint: disable=redefined-builtin
    ret = fn(_ctypes.byref(bytes), device)
    _cudaCheckReturn(ret)
    return bytes.value


def is_available() -> bool:
    """Whether there are any CUDA visible devices."""

    try:
        return cuDeviceGetCount() > 0
    except CUDAError:
        return False
