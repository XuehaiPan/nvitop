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
"""The live classes for process running on the host and the GPU devices."""

# pylint: disable=too-many-lines

from __future__ import annotations

import contextlib
import datetime
import functools
import os
import threading
from abc import ABCMeta
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable
from weakref import WeakValueDictionary

from nvitop.api import host, libnvml
from nvitop.api.utils import (
    NA,
    UINT_MAX,
    NaType,
    Snapshot,
    bytes2human,
    memoize_when_activated,
    timedelta2human,
)


if TYPE_CHECKING:
    from typing_extensions import Self  # Python 3.11+

    from nvitop.api.device import Device


__all__ = ['HostProcess', 'GpuProcess', 'command_join']


if host.POSIX:

    def add_quotes(s: str) -> str:
        """Return a shell-escaped version of the string."""
        if s == '':
            return '""'
        if '$' not in s and '\\' not in s and '\n' not in s:
            if ' ' not in s:
                return s
            if '"' not in s:
                return f'"{s}"'
        if "'" not in s and '\n' not in s:
            return f"'{s}'"
        return '"{}"'.format(
            s.replace('\\', r'\\').replace('"', r'\"').replace('$', r'\$').replace('\n', r'\n'),
        )

elif host.WINDOWS:

    def add_quotes(s: str) -> str:
        """Return a shell-escaped version of the string."""
        if s == '':
            return '""'
        if '%' not in s and '^' not in s and '\n' not in s:
            if ' ' not in s:
                return s
            if '"' not in s:
                return f'"{s}"'
        return '"{}"'.format(
            s.replace('^', '^^').replace('"', '^"').replace('%', '^%').replace('\n', r'\n'),
        )

else:

    def add_quotes(s: str) -> str:
        """Return a shell-escaped version of the string."""
        return '"{}"'.format(s.replace('\n', r'\n'))


def command_join(cmdline: list[str]) -> str:
    """Return a shell-escaped string from command line arguments."""
    if len(cmdline) == 1 and not (
        # May be modified by `setproctitle`
        os.path.isfile(cmdline[0])
        and os.path.isabs(cmdline[0])
    ):
        return cmdline[0]
    return ' '.join(map(add_quotes, cmdline))


_RAISE = object()
_USE_FALLBACK_WHEN_RAISE = threading.local()  # see also `GpuProcess.failsafe`


def auto_garbage_clean(
    fallback: Any = _RAISE,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Remove the object references in the instance cache if the method call fails (the process is gone).

    The fallback value will be used with `:meth:`GpuProcess.failsafe`` context manager, otherwise
    raises an exception when falls.
    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapped(self: GpuProcess, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, *args, **kwargs)
            except host.PsutilError as ex:
                try:
                    with GpuProcess.INSTANCE_LOCK:
                        del GpuProcess.INSTANCES[(self.pid, self.device)]
                except (KeyError, AttributeError):
                    pass
                try:
                    with HostProcess.INSTANCE_LOCK:
                        del HostProcess.INSTANCES[self.pid]
                except KeyError:
                    pass
                # See also `GpuProcess.failsafe`
                if fallback is _RAISE or not getattr(_USE_FALLBACK_WHEN_RAISE, 'value', False):
                    raise
                if isinstance(fallback, tuple):
                    if isinstance(ex, host.AccessDenied) and fallback == ('No Such Process',):
                        return ['No Permissions']
                    return list(fallback)
                return fallback

        return wrapped

    return wrapper


class HostProcess(host.Process, metaclass=ABCMeta):
    """Represent an OS process with the given PID.

    If PID is omitted current process PID (:func:`os.getpid`) is used. The instance will be cache
    during the lifetime of the process.

    Examples:
        >>> HostProcess()  # the current process
        HostProcess(pid=12345, name='python3', status='running', started='00:55:43')

        >>> p1 = HostProcess(12345)
        >>> p2 = HostProcess(12345)
        >>> p1 is p2                 # the same instance
        True

        >>> import copy
        >>> copy.deepcopy(p1) is p1  # the same instance
        True

        >>> p = HostProcess(pid=12345)
        >>> p.cmdline()
        ['python3', '-c', 'import IPython; IPython.terminal.ipapp.launch_new_instance()']
        >>> p.command()  # the result is in shell-escaped format
        'python3 -c "import IPython; IPython.terminal.ipapp.launch_new_instance()"'

        >>> p.as_snapshot()
        HostProcessSnapshot(
            real=HostProcess(pid=12345, name='python3', status='running', started='00:55:43'),
            cmdline=['python3', '-c', 'import IPython; IPython.terminal.ipapp.launch_new_instance()'],
            command='python3 -c "import IPython; IPython.terminal.ipapp.launch_new_instance()"',
            connections=[],
            cpu_percent=0.3,
            cpu_times=pcputimes(user=2.180019456, system=0.18424464, children_user=0.0, children_system=0.0),
            create_time=1656608143.31,
            cwd='/home/panxuehai',
            environ={...},
            ...
        )
    """

    INSTANCE_LOCK: threading.RLock = threading.RLock()
    INSTANCES: WeakValueDictionary[int, HostProcess] = WeakValueDictionary()

    _pid: int
    _super_gone: bool
    _username: str | None
    _ident: tuple
    _lock: threading.RLock

    def __new__(cls, pid: int | None = None) -> Self:
        """Return the cached instance of :class:`HostProcess`."""
        if pid is None:
            pid = os.getpid()

        with cls.INSTANCE_LOCK:
            try:
                instance = cls.INSTANCES[pid]
                if instance.is_running():
                    return instance
            except KeyError:
                pass

            instance = super().__new__(cls)

            instance._super_gone = False
            instance._username = None
            host.Process._init(instance, pid, True)
            try:
                host.Process.cpu_percent(instance)
            except host.PsutilError:
                pass

            cls.INSTANCES[pid] = instance

            return instance

    # pylint: disable-next=unused-argument,super-init-not-called
    def __init__(self, pid: int | None = None) -> None:
        """Initialize the instance."""

    @property
    def _gone(self) -> bool:
        return self._super_gone

    @_gone.setter
    def _gone(self, value: bool) -> None:
        if value:
            with self.INSTANCE_LOCK:
                self.INSTANCES.pop(self.pid, None)
        self._super_gone = value

    def __repr__(self) -> str:
        """Return a string representation of the process."""
        return super().__repr__().replace(self.__class__.__module__ + '.', '', 1)

    def __reduce__(self) -> tuple[type[HostProcess], tuple[int]]:
        """Return state information for pickling."""
        return self.__class__, (self.pid,)

    if host.WINDOWS:

        def username(self) -> str:
            """The name of the user that owns the process.

            On Windows, the domain name will be removed if it is present.

            Raises:
                host.NoSuchProcess:
                    If the process is gone.
                host.AccessDenied:
                    If the user do not have read privilege to the process' status file.
            """
            if self._username is None:  # pylint: disable=access-member-before-definition
                self._username = (  # pylint: disable=attribute-defined-outside-init
                    super().username().split('\\')[-1]
                )
            return self._username

    else:

        def username(self) -> str:
            """The name of the user that owns the process.

            On UNIX this is calculated by using *real* process uid.

            Raises:
                host.NoSuchProcess:
                    If the process is gone.
                host.AccessDenied:
                    If the user do not have read privilege to the process' status file.
            """
            if self._username is None:  # pylint: disable=access-member-before-definition
                self._username = (  # pylint: disable=attribute-defined-outside-init
                    super().username()
                )
            return self._username

    @memoize_when_activated
    def cmdline(self) -> list[str]:
        """The command line this process has been called with.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        cmdline = super().cmdline()
        if len(cmdline) > 1:
            cmdline = '\0'.join(cmdline).rstrip('\0').split('\0')
        return cmdline

    def command(self) -> str:
        """Return a shell-escaped string from command line arguments.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        return command_join(self.cmdline())

    @memoize_when_activated
    def running_time(self) -> datetime.timedelta:
        """The elapsed time this process has been running in :class:`datetime.timedelta`.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        return datetime.datetime.now() - datetime.datetime.fromtimestamp(self.create_time())

    def running_time_human(self) -> str:
        """The elapsed time this process has been running in human readable format.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        return timedelta2human(self.running_time())

    def running_time_in_seconds(self) -> float:  # in seconds
        """The elapsed time this process has been running in seconds.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        return self.running_time().total_seconds()

    elapsed_time = running_time
    elapsed_time_human = running_time_human
    elapsed_time_in_seconds = running_time_in_seconds

    def rss_memory(self) -> int:  # in bytes
        """The used resident set size (RSS) memory of the process in bytes.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        return self.memory_info().rss

    def parent(self) -> HostProcess | None:
        """Return the parent process as a :class:`HostProcess` instance or :data:`None` if there is no parent.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        parent = super().parent()
        if parent is not None:
            return HostProcess(parent.pid)
        return None

    def children(self, recursive: bool = False) -> list[HostProcess]:
        """Return the children of this process as a list of :class:`HostProcess` instances.

        If *recursive* is :data:`True` return all the descendants.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        return [HostProcess(child.pid) for child in super().children(recursive)]

    @contextlib.contextmanager
    def oneshot(self) -> Generator[None]:
        """A utility context manager which considerably speeds up the retrieval of multiple process information at the same time.

        Internally different process info (e.g. name, ppid, uids, gids, ...) may be fetched by using
        the same routine, but only one information is returned and the others are discarded. When
        using this context manager the internal routine is executed once (in the example below on
        ``name()``) and the other info are cached.

        The cache is cleared when exiting the context manager block. The advice is to use this every
        time you retrieve more than one information about the process.

        Examples:
            >>> from nvitop import HostProcess
            >>> p = HostProcess()
            >>> with p.oneshot():
            ...     p.name()         # collect multiple info
            ...     p.cpu_times()    # return cached value
            ...     p.cpu_percent()  # return cached value
            ...     p.create_time()  # return cached value
        """  # pylint: disable=line-too-long
        with self._lock:
            if hasattr(self, '_cache'):
                yield
            else:
                with super().oneshot():
                    # pylint: disable=no-member
                    try:
                        self.cmdline.cache_activate(self)  # type: ignore[attr-defined]
                        self.running_time.cache_activate(self)  # type: ignore[attr-defined]
                        yield
                    finally:
                        self.cmdline.cache_deactivate(self)  # type: ignore[attr-defined]
                        self.running_time.cache_deactivate(self)  # type: ignore[attr-defined]

    def as_snapshot(
        self,
        attrs: Iterable[str] | None = None,
        ad_value: Any | None = None,
    ) -> Snapshot:
        """Return a onetime snapshot of the process."""
        with self.oneshot():
            attributes = self.as_dict(attrs=attrs, ad_value=ad_value)

            if attrs is None:
                for attr in ('command', 'running_time', 'running_time_human'):
                    try:
                        attributes[attr] = getattr(self, attr)()
                    except (host.AccessDenied, host.ZombieProcess):  # noqa: PERF203
                        attributes[attr] = ad_value

        return Snapshot(real=self, **attributes)


@HostProcess.register
class GpuProcess:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Represent a process with the given PID running on the given GPU device.

    The instance will be cache during the lifetime of the process.

    The same host process can use multiple GPU devices. The :class:`GpuProcess` instances
    representing the same PID on the host but different GPU devices are different.
    """

    INSTANCE_LOCK: threading.RLock = threading.RLock()
    INSTANCES: WeakValueDictionary[tuple[int, Device], GpuProcess] = WeakValueDictionary()

    _pid: int
    _host: HostProcess
    _device: Device
    _username: str | None
    _ident: tuple
    _hash: int | None

    # pylint: disable-next=too-many-arguments
    def __new__(
        cls,
        pid: int | None,
        device: Device,
        *,
        # pylint: disable=unused-argument
        gpu_memory: int | NaType | None = None,
        gpu_instance_id: int | NaType | None = None,
        compute_instance_id: int | NaType | None = None,
        type: str | NaType | None = None,  # pylint: disable=redefined-builtin
        # pylint: enable=unused-argument
    ) -> Self:
        """Return the cached instance of :class:`GpuProcess`."""
        if pid is None:
            pid = os.getpid()

        with cls.INSTANCE_LOCK:
            try:
                instance = cls.INSTANCES[(pid, device)]
                if instance.is_running():
                    return instance  # type: ignore[return-value]
            except KeyError:
                pass

            instance = super().__new__(cls)

            instance._pid = pid
            instance._host = HostProcess(pid)
            instance._ident = (*instance._host._ident, device.index)
            instance._device = device

            instance._hash = None
            instance._username = None

            cls.INSTANCES[(pid, device)] = instance

            return instance

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        pid: int | None,  # pylint: disable=unused-argument
        device: Device,
        *,
        gpu_memory: int | NaType | None = None,
        gpu_instance_id: int | NaType | None = None,
        compute_instance_id: int | NaType | None = None,
        type: str | NaType | None = None,  # pylint: disable=redefined-builtin
    ) -> None:
        """Initialize the instance returned by :meth:`__new__()`."""
        if gpu_memory is None and not hasattr(self, '_gpu_memory'):
            gpu_memory = NA
        if gpu_memory is not None:
            self.set_gpu_memory(gpu_memory)

        if type is None and not hasattr(self, '_type'):
            type = NA
        if type is not None:
            self.type = type

        if gpu_instance_id is not None and compute_instance_id is not None:
            self._gpu_instance_id = gpu_instance_id if gpu_instance_id != UINT_MAX else NA
            self._compute_instance_id = (
                compute_instance_id if compute_instance_id != UINT_MAX else NA
            )
        elif device.is_mig_device():
            self._gpu_instance_id = device.gpu_instance_id()
            self._compute_instance_id = device.compute_instance_id()
        else:
            self._gpu_instance_id = self._compute_instance_id = NA

        for util in ('sm', 'memory', 'encoder', 'decoder'):
            if not hasattr(self, f'_gpu_{util}_utilization'):
                setattr(self, f'_gpu_{util}_utilization', NA)

    def __repr__(self) -> str:
        """Return a string representation of the GPU process."""
        return '{}(pid={}, gpu_memory={}, type={}, device={}, host={})'.format(  # noqa: UP032
            self.__class__.__name__,
            self.pid,
            self.gpu_memory_human(),
            self.type,
            self.device,
            self.host,
        )

    def __eq__(self, other: object) -> bool:
        """Test equality to other object."""
        if not isinstance(other, (GpuProcess, host.Process)):
            return NotImplemented
        return self._ident == other._ident

    def __hash__(self) -> int:
        """Return a hash value of the GPU process."""
        if self._hash is None:  # pylint: disable=access-member-before-definition
            self._hash = hash(self._ident)  # pylint: disable=attribute-defined-outside-init
        return self._hash

    def __getattr__(self, name: str) -> Any | Callable[..., Any]:
        """Get a member from the instance or fallback to the host process instance if missing.

        Raises:
            AttributeError:
                If the attribute is not defined in either :class:`GpuProcess` nor :class:`HostProcess`.
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """
        try:
            return super().__getattr__(name)  # type: ignore[misc]
        except AttributeError:
            if name == '_cache':
                raise
            attribute = getattr(self.host, name)
            if isinstance(attribute, FunctionType):
                attribute = auto_garbage_clean(fallback=_RAISE)(attribute)

            setattr(self, name, attribute)
            return attribute

    @property
    def pid(self) -> int:
        """The process PID."""
        return self._pid

    @property
    def host(self) -> HostProcess:
        """The process instance running on the host."""
        return self._host

    @property
    def device(self) -> Device:
        """The GPU device the process running on.

        The same host process can use multiple GPU devices. The :class:`GpuProcess` instances
        representing the same PID on the host but different GPU devices are different.
        """
        return self._device

    def gpu_instance_id(self) -> int | NaType:
        """The GPU instance ID of the MIG device, or :const:`nvitop.NA` if not applicable."""
        return self._gpu_instance_id

    def compute_instance_id(self) -> int | NaType:
        """The compute instance ID of the MIG device, or :const:`nvitop.NA` if not applicable."""
        return self._compute_instance_id

    def gpu_memory(self) -> int | NaType:  # in bytes
        """The used GPU memory in bytes, or :const:`nvitop.NA` if not applicable."""
        return self._gpu_memory

    def gpu_memory_human(self) -> str | NaType:  # in human readable
        """The used GPU memory in human readable format, or :const:`nvitop.NA` if not applicable."""
        return self._gpu_memory_human

    def gpu_memory_percent(self) -> float | NaType:  # in percentage
        """The percentage of used GPU memory by the process, or :const:`nvitop.NA` if not applicable."""
        return self._gpu_memory_percent

    def gpu_sm_utilization(self) -> int | NaType:  # in percentage
        """The utilization rate of SM (Streaming Multiprocessor), or :const:`nvitop.NA` if not applicable."""
        return self._gpu_sm_utilization

    def gpu_memory_utilization(self) -> int | NaType:  # in percentage
        """The utilization rate of GPU memory bandwidth, or :const:`nvitop.NA` if not applicable."""
        return self._gpu_memory_utilization

    def gpu_encoder_utilization(self) -> int | NaType:  # in percentage
        """The utilization rate of the encoder, or :const:`nvitop.NA` if not applicable."""
        return self._gpu_encoder_utilization

    def gpu_decoder_utilization(self) -> int | NaType:  # in percentage
        """The utilization rate of the decoder, or :const:`nvitop.NA` if not applicable."""
        return self._gpu_decoder_utilization

    def set_gpu_memory(self, value: int | NaType) -> None:
        """Set the used GPU memory in bytes."""
        # pylint: disable=attribute-defined-outside-init
        self._gpu_memory = memory_used = value
        self._gpu_memory_human = bytes2human(self.gpu_memory())
        memory_total = self.device.memory_total()
        gpu_memory_percent = NA
        if libnvml.nvmlCheckReturn(memory_used, int) and libnvml.nvmlCheckReturn(memory_total, int):
            gpu_memory_percent = round(100.0 * memory_used / memory_total, 1)  # type: ignore[assignment]
        self._gpu_memory_percent = gpu_memory_percent

    def set_gpu_utilization(
        self,
        gpu_sm_utilization: int | NaType | None = None,
        gpu_memory_utilization: int | NaType | None = None,
        gpu_encoder_utilization: int | NaType | None = None,
        gpu_decoder_utilization: int | NaType | None = None,
    ) -> None:
        """Set the GPU utilization rates."""
        # pylint: disable=attribute-defined-outside-init
        if gpu_sm_utilization is not None:
            self._gpu_sm_utilization = gpu_sm_utilization
        if gpu_memory_utilization is not None:
            self._gpu_memory_utilization = gpu_memory_utilization
        if gpu_encoder_utilization is not None:
            self._gpu_encoder_utilization = gpu_encoder_utilization
        if gpu_decoder_utilization is not None:
            self._gpu_decoder_utilization = gpu_decoder_utilization

    def update_gpu_status(self) -> int | NaType:
        """Update the GPU consumption status from a new NVML query."""
        self.set_gpu_memory(NA)
        self.set_gpu_utilization(NA, NA, NA, NA)
        processes = self.device.processes()
        process = processes.get(self.pid, self)
        if process is not self:
            # The current process is gone and the instance has been removed from the cache.
            # Update GPU status from the new instance.
            self.set_gpu_memory(process.gpu_memory())
            self.set_gpu_utilization(
                process.gpu_sm_utilization(),
                process.gpu_memory_utilization(),
                process.gpu_encoder_utilization(),
                process.gpu_decoder_utilization(),
            )
        return self.gpu_memory()

    @property
    def type(self) -> str | NaType:
        """The type of the GPU context.

        The type is one of the following:
            - :data:`'C'`: compute context
            - :data:`'G'`: graphics context
            - :data:`'C+G'`: both compute context and graphics context
            - :data:`'N/A'`: not applicable
        """
        return self._type

    @type.setter
    def type(self, value: str | NaType) -> None:
        if 'C' in value and 'G' in value:
            self._type = 'C+G'
        elif 'C' in value:
            self._type = 'C'
        elif 'G' in value:
            self._type = 'G'
        else:
            self._type = NA

    @auto_garbage_clean(fallback=False)
    def is_running(self) -> bool:
        """Return whether this process is running."""
        return self.host.is_running()

    @auto_garbage_clean(fallback='terminated')
    def status(self) -> str:
        """The process current status.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return self.host.status()

    @auto_garbage_clean(fallback=NA)
    def create_time(self) -> float | NaType:
        """The process creation time as a floating point number expressed in seconds since the epoch.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return self.host.create_time()

    @auto_garbage_clean(fallback=NA)
    def running_time(self) -> datetime.timedelta | NaType:
        """The elapsed time this process has been running in :class:`datetime.timedelta`.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return self.host.running_time()

    def running_time_human(self) -> str | NaType:
        """The elapsed time this process has been running in human readable format.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return timedelta2human(self.running_time())

    def running_time_in_seconds(self) -> float | NaType:
        """The elapsed time this process has been running in seconds.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        running_time = self.running_time()
        if running_time is NA:
            return NA
        return running_time.total_seconds()

    elapsed_time = running_time
    elapsed_time_human = running_time_human
    elapsed_time_in_seconds = running_time_in_seconds

    @auto_garbage_clean(fallback=NA)
    def username(self) -> str | NaType:
        """The name of the user that owns the process.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        if self._username is None:  # pylint: disable=access-member-before-definition
            self._username = self.host.username()  # pylint: disable=attribute-defined-outside-init
        return self._username

    @auto_garbage_clean(fallback=NA)
    def name(self) -> str | NaType:
        """The process name.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return self.host.name()

    @auto_garbage_clean(fallback=NA)
    def cpu_percent(self) -> float | NaType:  # in percentage
        """Return a float representing the current process CPU utilization as a percentage.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return self.host.cpu_percent()

    @auto_garbage_clean(fallback=NA)
    def memory_percent(self) -> float | NaType:  # in percentage
        """Compare process RSS memory to total physical system memory and calculate process memory utilization as a percentage.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """  # pylint: disable=line-too-long
        return self.host.memory_percent()

    host_memory_percent = memory_percent  # in percentage

    @auto_garbage_clean(fallback=NA)
    def host_memory(self) -> int | NaType:  # in bytes
        """The used resident set size (RSS) memory of the process in bytes.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return self.host.rss_memory()

    def host_memory_human(self) -> str | NaType:
        """The used resident set size (RSS) memory of the process in human readable format.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return bytes2human(self.host_memory())

    rss_memory = host_memory  # in bytes

    # For `AccessDenied` error the fallback value is `['No Permissions']`
    @auto_garbage_clean(fallback=('No Such Process',))
    def cmdline(self) -> list[str]:
        """The command line this process has been called with.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        cmdline = self.host.cmdline()
        if len(cmdline) == 0 and not self._gone:
            cmdline = ['Zombie Process']
        return cmdline

    def command(self) -> str:
        """Return a shell-escaped string from command line arguments.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        return command_join(self.cmdline())

    @auto_garbage_clean(fallback=_RAISE)
    def host_snapshot(self) -> Snapshot:
        """Return a onetime snapshot of the host process."""
        with self.host.oneshot():
            return Snapshot(
                real=self.host,
                is_running=self.is_running(),
                status=self.status(),
                username=self.username(),
                name=self.name(),
                cmdline=self.cmdline(),
                command=self.command(),
                cpu_percent=self.cpu_percent(),
                memory_percent=self.memory_percent(),
                host_memory=self.host_memory(),
                host_memory_human=self.host_memory_human(),
                running_time=self.running_time(),
                running_time_human=self.running_time_human(),
                running_time_in_seconds=self.running_time_in_seconds(),
            )

    @auto_garbage_clean(fallback=_RAISE)
    def as_snapshot(
        self,
        *,
        host_process_snapshot_cache: dict[int, Snapshot] | None = None,
    ) -> Snapshot:
        """Return a onetime snapshot of the process on the GPU device.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. Also, consider using the batched version to take
            snapshots with :meth:`GpuProcess.take_snapshots`, which caches the results and reduces
            redundant queries. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """
        host_process_snapshot_cache = host_process_snapshot_cache or {}
        try:
            host_snapshot = host_process_snapshot_cache[self.pid]
        except KeyError:
            host_snapshot = host_process_snapshot_cache[self.pid] = self.host_snapshot()

        return Snapshot(
            real=self,
            pid=self.pid,
            # host
            host=host_snapshot,
            is_running=host_snapshot.is_running,
            status=host_snapshot.status,
            username=host_snapshot.username,
            name=host_snapshot.name,
            cmdline=host_snapshot.cmdline,
            command=host_snapshot.command,
            cpu_percent=host_snapshot.cpu_percent,
            memory_percent=host_snapshot.memory_percent,
            host_memory=host_snapshot.host_memory,
            host_memory_human=host_snapshot.host_memory_human,
            running_time=host_snapshot.running_time,
            running_time_human=host_snapshot.running_time_human,
            running_time_in_seconds=host_snapshot.running_time_in_seconds,
            # device
            device=self.device,
            type=self.type,
            gpu_instance_id=self.gpu_instance_id(),
            compute_instance_id=self.compute_instance_id(),
            gpu_memory=self.gpu_memory(),
            gpu_memory_human=self.gpu_memory_human(),
            gpu_memory_percent=self.gpu_memory_percent(),
            gpu_sm_utilization=self.gpu_sm_utilization(),
            gpu_memory_utilization=self.gpu_memory_utilization(),
            gpu_encoder_utilization=self.gpu_encoder_utilization(),
            gpu_decoder_utilization=self.gpu_decoder_utilization(),
        )

    @classmethod
    def take_snapshots(  # batched version of `as_snapshot`
        cls,
        gpu_processes: Iterable[GpuProcess],
        *,
        failsafe: bool = False,
    ) -> list[Snapshot]:
        """Take snapshots for a list of :class:`GpuProcess` instances.

        If *failsafe* is :data:`True`, then if any method fails, the fallback value in
        :func:`auto_garbage_clean` will be used.
        """
        cache: dict[int, Snapshot] = {}
        context: Callable[[], contextlib.AbstractContextManager[None]] = (
            cls.failsafe if failsafe else contextlib.nullcontext
        )
        with context():
            return [
                process.as_snapshot(host_process_snapshot_cache=cache) for process in gpu_processes
            ]

    @classmethod
    @contextlib.contextmanager
    def failsafe(cls) -> Generator[None]:
        """A context manager that enables fallback values for methods that fail.

        Examples:
            >>> p = GpuProcess(pid=10000, device=Device(0))  # process does not exist
            >>> p
            GpuProcess(pid=10000, gpu_memory=N/A, type=N/A, device=PhysicalDevice(index=0, name="NVIDIA GeForce RTX 3070", total_memory=8192MiB), host=HostProcess(pid=10000, status='terminated'))
            >>> p.cpu_percent()
            Traceback (most recent call last):
                ...
            NoSuchProcess: process no longer exists (pid=10000)

            >>> # Failsafe to the fallback value instead of raising exceptions
            ... with GpuProcess.failsafe():
            ...     print('fallback:              {!r}'.format(p.cpu_percent()))
            ...     print('fallback (float cast): {!r}'.format(float(p.cpu_percent())))  # `nvitop.NA` can be cast to float or int
            ...     print('fallback (int cast):   {!r}'.format(int(p.cpu_percent())))    # `nvitop.NA` can be cast to float or int
            fallback:              'N/A'
            fallback (float cast): nan
            fallback (int cast):   0
        """  # pylint: disable=line-too-long
        global _USE_FALLBACK_WHEN_RAISE  # pylint: disable=global-statement,global-variable-not-assigned

        prev_value = getattr(_USE_FALLBACK_WHEN_RAISE, 'value', False)
        try:
            _USE_FALLBACK_WHEN_RAISE.value = True
            yield
        finally:
            _USE_FALLBACK_WHEN_RAISE.value = prev_value
