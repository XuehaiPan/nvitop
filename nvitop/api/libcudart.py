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
"""Python bindings for the `CUDA Runtime APIs <https://docs.nvidia.com/cuda/cuda-runtime-api>`_."""

# pylint: disable=invalid-name

from __future__ import annotations

import ctypes as _ctypes
import glob as _glob
import os as _os
import platform as _platform
import sys as _sys
import threading as _threading
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any
from typing import Callable as _Callable
from typing import ClassVar as _ClassVar


if _TYPE_CHECKING:
    from typing_extensions import Self as _Self  # Python 3.11+


_cudaError_t = _ctypes.c_int

# Error codes #
# pylint: disable=line-too-long
cudaSuccess = 0
"""The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete (see :func:`cudaEventQuery` and :func:`cudaStreamQuery`)."""
cudaErrorInvalidValue = 1
"""This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values."""
cudaErrorMemoryAllocation = 2
"""The API call failed because it was unable to allocate enough memory to perform the requested operation."""
cudaErrorInitializationError = 3
"""The API call failed because the CUDA driver and runtime could not be initialized."""
cudaErrorCudartUnloading = 4
"""This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded."""
cudaErrorProfilerDisabled = 5
"""This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler."""
cudaErrorInvalidConfiguration = 9
"""This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See cudaDeviceProp for more device limitations."""
cudaErrorInvalidPitchValue = 12
"""This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch."""
cudaErrorInvalidSymbol = 13
"""This indicates that the symbol name / identifier passed to the API call is not a valid name or identifier."""
cudaErrorInvalidTexture = 18
"""This indicates that the texture passed to the API call is not a valid texture."""
cudaErrorInvalidTextureBinding = 19
"""This indicates that the texture binding is not valid. This occurs if you call :func:`cudaGetTextureAlignmentOffset` with an unbound texture."""
cudaErrorInvalidChannelDescriptor = 20
"""This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by :data:`cudaChannelFormatKind`, or if one of the dimensions is invalid."""
cudaErrorInvalidMemcpyDirection = 21
"""This indicates that the direction of the :func:`memcpy` passed to the API call is not one of the types specified by :data:`cudaMemcpyKind`."""
cudaErrorInvalidFilterSetting = 26
"""This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA."""
cudaErrorInvalidNormSetting = 27
"""This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA."""
cudaErrorStubLibrary = 34
"""This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error."""
cudaErrorInsufficientDriver = 35
"""This indicates that the installed NVIDIA CUDA driver is older than the CUDA Runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run."""
cudaErrorCallRequiresNewerDriver = 36
"""This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed."""
cudaErrorInvalidSurface = 37
"""This indicates that the surface passed to the API call is not a valid surface."""
cudaErrorDuplicateVariableName = 43
"""This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name."""
cudaErrorDuplicateTextureName = 44
"""This indicates that multiple textures (across separate CUDA source files in the application) share the same string name."""
cudaErrorDuplicateSurfaceName = 45
"""This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name."""
cudaErrorDevicesUnavailable = 46
"""This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy / unavailable due to use of :data:`cudaComputeModeProhibited`, :data:`cudaComputeModeExclusiveProcess`, or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed."""
cudaErrorIncompatibleDriverContext = 49
"""This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are using CUDA Runtime / Driver interoperability and have created an existing Driver context using the driver API. The Driver context may be incompatible either because the Driver context was created using an older version of the API, because the Runtime API call expects a primary driver context and the Driver context is not primary, or because the Driver context has been destroyed."""
cudaErrorMissingConfiguration = 52
"""The device function being invoked (usually via :func:`cudaLaunchKernel`) was not previously configured via the :func:`cudaConfigureCall` function."""
cudaErrorLaunchMaxDepthExceeded = 65
"""This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches."""
cudaErrorLaunchFileScopedTex = 66
"""This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's."""
cudaErrorLaunchFileScopedSurf = 67
"""This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's."""
cudaErrorSyncDepthExceeded = 68
"""This error indicates that a call to :func:`cudaDeviceSynchronize` made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levels of grids) or user specified device limit :data:`cudaLimitDevRuntimeSyncDepth`. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which :func:`cudaDeviceSynchronize` will be called must be specified with the :data:`cudaLimitDevRuntimeSyncDepth` limit to the :func:`cudaDeviceSetLimit` api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory that cannot be used for user allocations. Note that :func:`cudaDeviceSynchronize` made from device runtime is only supported on devices of compute capability < 9.0."""
cudaErrorLaunchPendingCountExceeded = 69
"""This error indicates that a device runtime grid launch failed because the launch would exceed the limit :data:`cudaLimitDevRuntimePendingLaunchCount`. For this launch to proceed successfully, :func:`cudaDeviceSetLimit` must be called to set the :data:`cudaLimitDevRuntimePendingLaunchCount` to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raising the limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for user allocations."""
cudaErrorInvalidDeviceFunction = 98
"""The requested device function does not exist or is not compiled for the proper device architecture."""
cudaErrorNoDevice = 100
"""This indicates that no CUDA-capable devices were detected by the installed CUDA driver."""
cudaErrorInvalidDevice = 101
"""This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device."""
cudaErrorDeviceNotLicensed = 102
"""This indicates that the device doesn't have a valid Grid License."""
cudaErrorSoftwareValidityNotEstablished = 103
"""By default, the CUDA Runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validity of either the runtime or the driver could not be established."""
cudaErrorStartupFailure = 127
"""This indicates an internal startup failure in the CUDA Runtime."""
cudaErrorInvalidKernelImage = 200
"""This indicates that the device kernel image is invalid."""
cudaErrorDeviceUninitialized = 201
"""This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had :func`cuCtxDestroy` invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls)."""
cudaErrorMapBufferObjectFailed = 205
"""This indicates that the buffer object could not be mapped."""
cudaErrorUnmapBufferObjectFailed = 206
"""This indicates that the buffer object could not be unmapped."""
cudaErrorArrayIsMapped = 207
"""This indicates that the specified array is currently mapped and thus cannot be destroyed."""
cudaErrorAlreadyMapped = 208
"""This indicates that the resource is already mapped."""
cudaErrorNoKernelImageForDevice = 209
"""This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration."""
cudaErrorAlreadyAcquired = 210
"""This indicates that a resource has already been acquired."""
cudaErrorNotMapped = 211
"""This indicates that a resource is not mapped."""
cudaErrorNotMappedAsArray = 212
"""This indicates that a mapped resource is not available for access as an array."""
cudaErrorNotMappedAsPointer = 213
"""This indicates that a mapped resource is not available for access as a pointer."""
cudaErrorECCUncorrectable = 214
"""This indicates that an uncorrectable ECC error was detected during execution."""
cudaErrorUnsupportedLimit = 215
"""This indicates that the :class:`cudaLimit` passed to the API call is not supported by the active device."""
cudaErrorDeviceAlreadyInUse = 216
"""This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread."""
cudaErrorPeerAccessUnsupported = 217
"""This error indicates that P2P access is not supported across the given devices."""
cudaErrorInvalidPtx = 218
"""A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device."""
cudaErrorInvalidGraphicsContext = 219
"""This indicates an error with the OpenGL or DirectX context."""
cudaErrorNvlinkUncorrectable = 220
"""This indicates that an uncorrectable NVLink error was detected during the execution."""
cudaErrorJitCompilerNotFound = 221
"""This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device."""
cudaErrorUnsupportedPtxVersion = 222
"""This indicates that the provided PTX was compiled with an unsupported toolchain. The most common reason for this, is the PTX was generated by a compiler newer than what is supported by the CUDA driver and PTX JIT compiler."""
cudaErrorJitCompilationDisabled = 223
"""This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device."""
cudaErrorUnsupportedExecAffinity = 224
"""This indicates that the provided execution affinity is not supported by the device."""
cudaErrorInvalidSource = 300
"""This indicates that the device kernel source is invalid."""
cudaErrorFileNotFound = 301
"""This indicates that the file specified was not found."""
cudaErrorSharedObjectSymbolNotFound = 302
"""This indicates that a link to a shared object failed to resolve."""
cudaErrorSharedObjectInitFailed = 303
"""This indicates that initialization of a shared object failed."""
cudaErrorOperatingSystem = 304
"""This error indicates that an OS call failed."""
cudaErrorInvalidResourceHandle = 400
"""This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like :data:`cudaStream_t` and :data:`cudaEvent_t`."""
cudaErrorIllegalState = 401
"""This indicates that a resource required by the API call is not in a valid state to perform the requested operation."""
cudaErrorSymbolNotFound = 500
"""This indicates that a named symbol was not found. Examples of symbols are global / constant variable names, driver function names, texture names, and surface names."""
cudaErrorNotReady = 600
"""This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than :data:`cudaSuccess` (which indicates completion). Calls that may return this value include :func:`cudaEventQuery` and :func:`cudaStreamQuery`."""
cudaErrorIllegalAddress = 700
"""The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorLaunchOutOfResources = 701
"""This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to :data:`cudaErrorInvalidConfiguration`, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count."""
cudaErrorLaunchTimeout = 702
"""This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device property kernelExecTimeoutEnabled for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorLaunchIncompatibleTexturing = 703
"""This error indicates a kernel launch that uses an incompatible texturing mode."""
cudaErrorPeerAccessAlreadyEnabled = 704
"""This error indicates that a call to :func:`cudaDeviceEnablePeerAccess` is trying to re-enable peer addressing on from a context which has already had peer addressing enabled."""
cudaErrorPeerAccessNotEnabled = 705
"""This error indicates that :func:`cudaDeviceDisablePeerAccess` is trying to disable peer addressing which has not been enabled yet via :func:`cudaDeviceEnablePeerAccess`."""
cudaErrorSetOnActiveProcess = 708
"""This indicates that the user has called :func:`cudaSetValidDevices`, :func:`cudaSetDeviceFlags`, :func:`cudaD3D9SetDirect3DDevice`, :func:`cudaD3D10SetDirect3DDevice`, :func:`cudaD3D11SetDirect3DDevice`, or :func:`cudaVDPAUSetVDPAUDevice` after initializing the CUDA Runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime / driver interoperability and there is an existing :class:`CUcontext` active on the host thread."""
cudaErrorContextIsDestroyed = 709
"""This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized."""
cudaErrorAssert = 710
"""An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorTooManyPeers = 711
"""This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to :func:`cudaEnablePeerAccess`."""
cudaErrorHostMemoryAlreadyRegistered = 712
"""This error indicates that the memory range passed to :func:`cudaHostRegister` has already been registered."""
cudaErrorHostMemoryNotRegistered = 713
"""This error indicates that the pointer passed to :func:`cudaHostUnregister` does not correspond to any currently registered memory region."""
cudaErrorHardwareStackError = 714
"""Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorIllegalInstruction = 715
"""The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorMisalignedAddress = 716
"""The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorInvalidAddressSpace = 717
"""While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorInvalidPc = 718
"""The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorLaunchFailure = 719
"""An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorCooperativeLaunchTooLarge = 720
"""This error indicates that the number of blocks launched per grid for a kernel that was launched via either :func:`cudaLaunchCooperativeKernel` or :func:`cudaLaunchCooperativeKernelMultiDevice` exceeds the maximum number of blocks as allowed by :func:`cudaOccupancyMaxActiveBlocksPerMultiprocessor` or :func:`cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` times the number of multiprocessors as specified by the device attribute :func:`cudaDevAttrMultiProcessorCount`."""
cudaErrorNotPermitted = 800
"""This error indicates the attempted operation is not permitted."""
cudaErrorNotSupported = 801
"""This error indicates the attempted operation is not supported on the current system or device."""
cudaErrorSystemNotReady = 802
"""This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide."""
cudaErrorSystemDriverMismatch = 803
"""This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions."""
cudaErrorCompatNotSupportedOnDevice = 804
"""This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the ``CUDA_VISIBLE_DEVICES`` environment variable."""
cudaErrorMpsConnectionFailed = 805
"""This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server."""
cudaErrorMpsRpcFailure = 806
"""This error indicates that the remote procedural call between the MPS server and the MPS client failed."""
cudaErrorMpsServerNotReady = 807
"""This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure."""
cudaErrorMpsMaxClientsReached = 808
"""This error indicates that the hardware resources required to create MPS client have been exhausted."""
cudaErrorMpsMaxConnectionsReached = 809
"""This error indicates the the hardware resources required to device connections have been exhausted."""
cudaErrorMpsClientTerminated = 810
"""This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorCdpNotSupported = 811
"""This error indicates, that the program is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it."""
cudaErrorCdpVersionMismatch = 812
"""This error indicates, that the program contains an unsupported interaction between different versions of CUDA Dynamic Parallelism."""
cudaErrorStreamCaptureUnsupported = 900
"""The operation is not permitted when the stream is capturing."""
cudaErrorStreamCaptureInvalidated = 901
"""The current capture sequence on the stream has been invalidated due to a previous error."""
cudaErrorStreamCaptureMerge = 902
"""The operation would have resulted in a merge of two independent capture sequences."""
cudaErrorStreamCaptureUnmatched = 903
"""The capture was not initiated in this stream."""
cudaErrorStreamCaptureUnjoined = 904
"""The capture sequence contains a fork that was not joined to the primary stream."""
cudaErrorStreamCaptureIsolation = 905
"""A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary."""
cudaErrorStreamCaptureImplicit = 906
"""The operation would have resulted in a disallowed implicit dependency on a current capture sequence from :data:`cudaStreamLegacy`."""
cudaErrorCapturedEvent = 907
"""The operation is not permitted on an event which was last recorded in a capturing stream."""
cudaErrorStreamCaptureWrongThread = 908
"""A stream capture sequence not initiated with the :data:`cudaStreamCaptureModeRelaxed` argument to :func:`cudaStreamBeginCapture` was passed to :func:`cudaStreamEndCapture` in a different thread."""
cudaErrorTimeout = 909
"""This indicates that the wait operation has timed out."""
cudaErrorGraphExecUpdateFailure = 910
"""This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update."""
cudaErrorExternalDevice = 911
"""This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched."""
cudaErrorInvalidClusterSize = 912
"""This indicates that a kernel launch error has occurred due to cluster misconfiguration."""
cudaErrorUnknown = 999
"""This indicates that an unknown internal error has occurred."""
# pylint: enable=line-too-long


# Error Checking #
class cudaError(Exception):
    """Base exception class for CUDA driver query errors."""

    _value_class_mapping: _ClassVar[dict[int, type[cudaError]]] = {}
    _errcode_to_string: _ClassVar[dict[int, str]] = {  # List of currently known error codes
        cudaErrorInitializationError:        'Initialization error.',
        cudaErrorSymbolNotFound:             'Named symbol not found.',
        cudaErrorInvalidValue:               'Invalid argument.',
        cudaErrorNoDevice:                   'No CUDA-capable device is detected.',
        cudaErrorInvalidDevice:              'Invalid device ordinal.',
        cudaErrorSystemDriverMismatch:       'System has unsupported display driver / CUDA driver combination.',
        cudaErrorCudartUnloading:            'Driver shutting down.',
        cudaErrorCompatNotSupportedOnDevice: 'Forward compatibility was attempted on non supported Hardware.',
        cudaErrorDeviceUninitialized:        'Invalid device context.',
    }  # fmt:skip
    _errcode_to_name: _ClassVar[dict[int, str]] = {}
    value: int

    def __new__(cls, value: int) -> _Self:
        """Map value to a proper subclass of :class:`cudaError`."""
        if cls is cudaError:
            # pylint: disable-next=self-cls-assignment
            cls = cudaError._value_class_mapping.get(value, cls)  # type: ignore[assignment]
        obj = Exception.__new__(cls)
        obj.value = value
        return obj

    def __repr__(self) -> str:
        """Return a string representation of the error."""
        # pylint: disable=no-member
        try:
            if self.value not in cudaError._errcode_to_string:
                cudaError._errcode_to_string[self.value] = '{}.'.format(
                    cuGetErrorString(self.value).rstrip('.').capitalize(),
                )
            if self.value not in cudaError._errcode_to_name:
                cudaError._errcode_to_name[self.value] = cudaGetErrorName(self.value)
            return (
                f'{cudaError._errcode_to_string[self.value]} '
                f'Code: {cudaError._errcode_to_name[self.value]} ({self.value}).'
            )
        except cudaError:
            return f'CUDA Error with code {self.value}.'

    def __eq__(self, other: object) -> bool:
        """Test equality to other object."""
        if not isinstance(other, cudaError):
            return NotImplemented
        return self.value == other.value  # pylint: disable=no-member

    def __reduce__(self) -> tuple[type[cudaError], tuple[int]]:
        """Return state information for pickling."""
        return cudaError, (self.value,)  # pylint: disable=no-member


def cudaExceptionClass(cudaErrorCode: int) -> type[cudaError]:
    """Map value to a proper subclass of :class:`cudaError`.

    Raises:
        ValueError: If the error code is not valid.
    """
    if cudaErrorCode not in cudaError._value_class_mapping:  # pylint: disable=protected-access
        raise ValueError(f'cudaErrorCode {cudaErrorCode} is not valid.')
    return cudaError._value_class_mapping[cudaErrorCode]  # pylint: disable=protected-access


def _extract_cuda_errors_as_classes() -> None:
    """Generate a hierarchy of classes on top of :class:`cudaError` class.

    Each CUDA Error gets a new :class:`cudaError` subclass. This way try-except blocks can filter
    appropriate exceptions more easily.

    :class:`cudaError` is a parent class. Each ``cudaError*`` gets it's own subclass.
    e.g. :data:`cudaErrorInvalidValue` will be turned into :class:`cudaError_InvalidValue`.
    """
    this_module = _sys.modules[__name__]
    cuda_error_names = [
        x
        for x in dir(this_module)
        if x.startswith('cudaError') and not x.startswith('cudaError_') and x != 'cudaError'
    ]
    for err_name in cuda_error_names:
        # e.g. Turn cudaErrorInvalidValue into cudaError_InvalidValue
        class_name = err_name.replace('cudaError', 'cudaError_')
        err_val = getattr(this_module, err_name)

        def gen_new(value: int) -> _Callable[[type[cudaError]], cudaError]:
            def new(cls: type[cudaError]) -> cudaError:
                return cudaError.__new__(cls, value)

            return new

        # pylint: disable=protected-access
        new_error_class = type(class_name, (cudaError,), {'__new__': gen_new(err_val)})
        new_error_class.__module__ = __name__
        if err_val in cudaError._errcode_to_string:
            new_error_class.__doc__ = (
                f'cudaError: {cudaError._errcode_to_string[err_val]} '
                f'Code: :data:`{err_name}` ({err_val}).'
            )
        else:
            new_error_class.__doc__ = f'CUDA Error with code :data:`{err_name}` ({err_val})'
        setattr(this_module, class_name, new_error_class)
        cudaError._value_class_mapping[err_val] = new_error_class
        cudaError._errcode_to_name[err_val] = err_name


# Add explicit references to appease linters
class __cudaError(cudaError):
    value: int

    def __new__(cls) -> cudaError:  # type: ignore[misc,empty-body]
        ...


cudaError_InitializationError: type[__cudaError]
cudaError_SymbolNotFound: type[__cudaError]
cudaError_InvalidValue: type[__cudaError]
cudaError_NoDevice: type[__cudaError]
cudaError_InvalidDevice: type[__cudaError]
cudaError_SystemDriverMismatch: type[__cudaError]
cudaError_CudartUnloading: type[__cudaError]
cudaError_CompatNotSupportedOnDevice: type[__cudaError]
cudaError_DeviceUninitialized: type[__cudaError]

_extract_cuda_errors_as_classes()
del _extract_cuda_errors_as_classes


def _cudaCheckReturn(ret: _Any) -> _Any:
    if ret != cudaSuccess:
        raise cudaError(ret)
    return ret


# Function access #
__cudaLib: _ctypes.CDLL | None = None
__libLoadLock: _threading.Lock = _threading.Lock()
# Function pointers are cached to prevent unnecessary libLoadLock locking
__cudaGetFunctionPointer_cache: dict[str, _ctypes._CFuncPtr] = {}  # type: ignore[name-defined]


def __cudaGetFunctionPointer(name: str) -> _ctypes._CFuncPtr:  # type: ignore[name-defined]
    """Get the function pointer from the CUDA Runtime library.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
        cudaError_SymbolNotFound:
            If cannot found the function pointer.
    """
    if name in __cudaGetFunctionPointer_cache:
        return __cudaGetFunctionPointer_cache[name]

    if __cudaLib is None:
        __LoadCudaLibrary()

    with __libLoadLock:
        try:
            __cudaGetFunctionPointer_cache[name] = getattr(__cudaLib, name)
            return __cudaGetFunctionPointer_cache[name]
        except AttributeError as ex:
            raise cudaError(cudaErrorSymbolNotFound) from ex


def __LoadCudaLibrary() -> None:  # pylint: disable=too-many-branches
    """Load the library if it isn't loaded already.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
    """
    global __cudaLib  # pylint: disable=global-statement

    if __cudaLib is None:
        # Lock to ensure only one caller loads the library
        with __libLoadLock:
            # Ensure the library still isn't loaded
            if __cudaLib is None:  # pylint: disable=too-many-nested-blocks
                # Platform specific libcudart location
                system = _platform.system()
                bits = _platform.architecture()[0].replace('bit', '')
                if system == 'Darwin':
                    lib_filenames = ['libcudart.dylib']
                elif system == 'Linux':
                    lib_filenames = ['libcudart.so']
                elif system == 'Windows':
                    lib_filenames = [f'cudart{bits}.dll', 'cudart.dll']
                else:
                    lib_filenames = []

                # Open library
                for lib_filename in lib_filenames:
                    try:
                        __cudaLib = _ctypes.CDLL(lib_filename)
                        break
                    except OSError:
                        pass

                # Try to load the library from the CUDA_PATH environment variable
                if __cudaLib is None:
                    cuda_paths = [
                        _os.getenv(env_name, '')
                        for env_name in ('CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT')
                    ]
                    if system != 'Windows':
                        cuda_paths.append('/usr/local/cuda')
                        candidate_paths = []
                        for cuda_path in cuda_paths:
                            if _os.path.isdir(cuda_path):
                                for lib_filename in lib_filenames:
                                    candidate_paths.extend(
                                        [
                                            _os.path.join(cuda_path, f'lib{bits}', lib_filename),
                                            _os.path.join(cuda_path, 'lib', lib_filename),
                                        ],
                                    )
                    else:
                        candidate_dirs = _os.getenv('PATH', '').split(_os.path.pathsep)
                        candidate_paths = []
                        for cuda_path in cuda_paths:
                            if _os.path.isdir(cuda_path):
                                candidate_dirs.extend(
                                    [
                                        _os.path.join(cuda_path, 'bin'),
                                        _os.path.join(cuda_path, f'lib{bits}'),
                                        _os.path.join(cuda_path, 'lib'),
                                    ],
                                )
                        for candidate_dir in candidate_dirs:
                            candidate_paths.extend(
                                _glob.iglob(_os.path.join(candidate_dir, f'cudart{bits}*.dll')),
                            )

                    # Normalize paths and remove duplicates
                    candidate_paths = list(
                        dict.fromkeys(
                            _os.path.normpath(_os.path.normcase(p)) for p in candidate_paths
                        ),
                    )
                    for lib_filename in candidate_paths:
                        try:
                            __cudaLib = _ctypes.CDLL(lib_filename)
                            break
                        except OSError:
                            pass

                if __cudaLib is None:
                    _cudaCheckReturn(cudaErrorInitializationError)


def cudaGetErrorName(error: int) -> str:
    """Get the string representation of an error code enum name.

    Returns: str
        A string containing the name of an error code in the enum. If the error code is not
        recognized, "unrecognized error code" is returned.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
    """
    fn = __cudaGetFunctionPointer('cudaGetErrorName')

    fn.restype = _ctypes.c_char_p  # otherwise return is an int
    p_name = fn(_cudaError_t(error))
    name = _ctypes.string_at(p_name)
    return name.decode('utf-8', errors='replace')


def cuGetErrorString(error: int) -> str:
    """Get the description string for an error code.

    Returns: str
        The description string for an error code. If the error code is not recognized, "unrecognized
        error code" is returned.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
    """
    fn = __cudaGetFunctionPointer('cudaGetErrorString')

    fn.restype = _ctypes.c_char_p  # otherwise return is an int
    p_name = fn(_cudaError_t(error))
    name = _ctypes.string_at(p_name)
    return name.decode('utf-8', errors='replace')


def cudaGetLastError() -> int:
    """Get the last error from a runtime call.

    Returns: int
        The last error that has been produced by any of the runtime calls in the same instance of
        the CUDA Runtime library in the host thread and resets it to :data:`cudaSuccess`.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
        cudaError_InsufficientDriver:
            If the installed NVIDIA CUDA driver is older than the CUDA Runtime library.
        cudaError_NoDevice:
            If no CUDA-capable devices were detected by the installed CUDA driver.
    """
    fn = __cudaGetFunctionPointer('cudaGetLastError')
    return fn()


def cudaPeekAtLastError() -> int:
    """Get the last error from a runtime call.

    Returns: int
        The last error that has been produced by any of the runtime calls in the same instance of
        the CUDA Runtime library in the host thread. This call does not reset the error to
        :data:`cudaSuccess` like :func:`cudaGetLastError`.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
        cudaError_InsufficientDriver:
            If the installed NVIDIA CUDA driver is older than the CUDA Runtime library.
        cudaError_NoDevice:
            If no CUDA-capable devices were detected by the installed CUDA driver.
    """
    fn = __cudaGetFunctionPointer('cudaPeekAtLastError')
    return fn()


def cudaDriverGetVersion() -> str:
    """Get the latest CUDA version supported by driver.

    Returns: str
        The latest version of CUDA supported by the driver of the form :data:`'<major>.<minor>'`.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
        cudaError_InsufficientDriver:
            If the installed NVIDIA CUDA driver is older than the CUDA Runtime library.
        cudaError_NoDevice:
            If no CUDA-capable devices were detected by the installed CUDA driver.
    """
    fn = __cudaGetFunctionPointer('cudaDriverGetVersion')

    driver_version = _ctypes.c_int()
    ret = fn(_ctypes.byref(driver_version))
    _cudaCheckReturn(ret)
    major = driver_version.value // 1000
    minor = (driver_version.value % 1000) // 10
    return f'{major}.{minor}'


def cudaRuntimeGetVersion() -> str:
    """Get the CUDA Runtime version.

    Returns: str
        The version number of the current CUDA Runtime instance of the form :data:`'<major>.<minor>'`.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
        cudaError_InsufficientDriver:
            If the installed NVIDIA CUDA driver is older than the CUDA Runtime library.
        cudaError_NoDevice:
            If no CUDA-capable devices were detected by the installed CUDA driver.
    """
    fn = __cudaGetFunctionPointer('cudaRuntimeGetVersion')

    runtime_version = _ctypes.c_int()
    ret = fn(_ctypes.byref(runtime_version))
    _cudaCheckReturn(ret)
    major = runtime_version.value // 1000
    minor = (runtime_version.value % 1000) // 10
    return f'{major}.{minor}'


def cudaGetDeviceCount() -> int:
    """Get the number of compute-capable devices.

    Returns: int
        The number of devices with compute capability greater or equal to 2.0 that are available for
        execution.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
        cudaError_InsufficientDriver:
            If the installed NVIDIA CUDA driver is older than the CUDA Runtime library.
        cudaError_NoDevice:
            If no CUDA-capable devices were detected by the installed CUDA driver.
    """
    fn = __cudaGetFunctionPointer('cudaGetDeviceCount')

    count = _ctypes.c_int(0)
    ret = fn(_ctypes.byref(count))
    _cudaCheckReturn(ret)
    return count.value


def cudaDeviceGetByPCIBusId(pciBusId: str) -> int:
    """Get a handle to a compute device.

    Args:
        pciBusId (str):
            String in one of the following forms: ``[domain]:[bus]:[device].[function]``,
            ``[domain]:[bus]:[device]``, ``[bus]:[device].[function]`` where ``domain``, ``bus``,
            ``device``, and ``function`` are all hexadecimal values.

    Returns: int
        A device ordinal given a PCI bus ID string.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
        cudaError_InsufficientDriver:
            If the installed NVIDIA CUDA driver is older than the CUDA Runtime library.
        cudaError_NoDevice:
            If no CUDA-capable devices were detected by the installed CUDA driver.
        cudaError_InvalidValue:
            If the value of :data:`pciBusId` is not a valid PCI bus identifier.
        cudaError_InvalidDevice:
            If the device ordinal supplied by the user does not correspond to a valid CUDA device.
    """
    fn = __cudaGetFunctionPointer('cudaDeviceGetByPCIBusId')

    device = _ctypes.c_int()
    ret = fn(_ctypes.byref(device), _ctypes.c_char_p(pciBusId.encode('utf-8')))
    _cudaCheckReturn(ret)
    return device.value


def cudaDeviceGetPCIBusId(device: int) -> str:
    """Get a PCI Bus Id string for the device.

    Returns: str
        An ASCII string identifying the device.

    Raises:
        cudaError_InitializationError:
            If cannot found the CUDA Runtime library.
        cudaError_InsufficientDriver:
            If the installed NVIDIA CUDA driver is older than the CUDA Runtime library.
        cudaError_NoDevice:
            If no CUDA-capable devices were detected by the installed CUDA driver.
        cudaError_InvalidValue:
            If the value of :data:`device` is not a valid device ordinal.
        cudaError_InvalidDevice:
            If the device ordinal supplied by the user does not correspond to a valid CUDA device.
    """
    fn = __cudaGetFunctionPointer('cudaDeviceGetPCIBusId')

    pciBusId = _ctypes.create_string_buffer(256)
    ret = fn(pciBusId, _ctypes.c_int(256), _ctypes.c_int(device))
    _cudaCheckReturn(ret)
    return pciBusId.value.decode('utf-8', errors='replace')


def is_available() -> bool:
    """Test whether there are any CUDA visible devices."""
    try:
        return cudaGetDeviceCount() > 0
    except cudaError:
        return False
