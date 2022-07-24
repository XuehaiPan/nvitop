# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""The live classes for process running on the host and the GPU devices."""

# pylint: disable=too-many-lines

import contextlib
import datetime
import functools
import os
import threading
from abc import ABCMeta
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from weakref import WeakValueDictionary

from nvitop.core import host, libnvml
from nvitop.core.utils import (
    NA,
    NaType,
    Snapshot,
    bytes2human,
    memoize_when_activated,
    timedelta2human,
)


if TYPE_CHECKING:
    from nvitop.core.device import Device


__all__ = ['HostProcess', 'GpuProcess', 'command_join']


if host.POSIX:

    def add_quotes(s: str) -> str:
        """Returns a shell-escaped version of the string."""

        if s == '':
            return '""'
        if '$' not in s and '\\' not in s and '\n' not in s:
            if ' ' not in s:
                return s
            if '"' not in s:
                return '"{}"'.format(s)
        if "'" not in s and '\n' not in s:
            return "'{}'".format(s)
        return '"{}"'.format(
            s.replace('\\', r'\\').replace('"', r'\"').replace('$', r'\$').replace('\n', r'\n')
        )

elif host.WINDOWS:

    def add_quotes(s: str) -> str:
        """Returns a shell-escaped version of the string."""

        if s == '':
            return '""'
        if '%' not in s and '^' not in s and '\n' not in s:
            if ' ' not in s:
                return s
            if '"' not in s:
                return '"{}"'.format(s)
        return '"{}"'.format(
            s.replace('^', '^^').replace('"', '^"').replace('%', '^%').replace('\n', r'\n')
        )

else:

    def add_quotes(s: str) -> str:
        """Returns a shell-escaped version of the string."""

        return '"{}"'.format(s.replace('\n', r'\n'))


def command_join(cmdline: List[str]) -> str:
    """Returns a shell-escaped string from command line arguments."""

    if len(cmdline) == 1 and not (
        # May be modified by `setproctitle`
        os.path.isfile(cmdline[0])
        and os.path.isabs(cmdline[0])
    ):
        return cmdline[0]
    return ' '.join(map(add_quotes, cmdline))


_RAISE = object()
_USE_FALLBACK_WHEN_RAISE = threading.local()  # see also `GpuProcess.failsafe`


def auto_garbage_clean(fallback=_RAISE):
    """Removes the object references in the instance cache if the method call fails (the process is gone).

    The fallback value will be used with `:meth:`GpuProcess.failsafe`` context manager, otherwise raises an
    exception when falls.
    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapped(self: 'GpuProcess', *args, **kwargs):
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
                if fallback is _RAISE or not getattr(
                    _USE_FALLBACK_WHEN_RAISE, 'value', False  # see also `GpuProcess.failsafe`
                ):
                    raise ex
                if isinstance(fallback, tuple):
                    if isinstance(ex, host.AccessDenied) and fallback == ('No Such Process',):
                        return ['No Permissions']
                    return list(fallback)
                return fallback

        return wrapped

    return wrapper


class HostProcess(host.Process, metaclass=ABCMeta):
    """Represents an OS process with the given PID.
    If PID is omitted current process PID (:func:`os.getpid`) is used.
    The instance will be cache during the lifetime of the process.

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

    INSTANCE_LOCK = threading.RLock()
    INSTANCES = WeakValueDictionary()

    def __new__(cls, pid: Optional[int] = None) -> 'HostProcess':
        """Returns the cached instance of :class:`HostProcess`."""

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
    def __init__(self, pid: Optional[int] = None) -> None:
        pass

    @property
    def _gone(self) -> bool:
        return self._super_gone

    @_gone.setter
    def _gone(self, value: bool) -> None:
        if value:
            with self.INSTANCE_LOCK:
                try:
                    del self.INSTANCES[self.pid]
                except KeyError:
                    pass
        self._super_gone = value

    def __str__(self) -> str:
        return super().__str__().replace(self.__class__.__module__ + '.', '', 1)

    __repr__ = __str__

    def __reduce__(self) -> Tuple[Type['HostProcess'], Tuple[int]]:
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
    def cmdline(self) -> List[str]:
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
        """Returns a shell-escaped string from command line arguments.

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

    def parent(self) -> Union['HostProcess', None]:
        """Returns the parent process as a :class:`HostProcess` instance. Returns :data:`None` if there is no parent.

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

    def children(self, recursive: bool = False) -> List['HostProcess']:
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
    def oneshot(self):
        """Utility context manager which considerably speeds up the retrieval of multiple process
        information at the same time.

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
        """

        with self._lock:
            if hasattr(self, '_cache'):
                yield
            else:
                with super().oneshot():
                    # pylint: disable=no-member
                    try:
                        self.cmdline.cache_activate(self)
                        self.running_time.cache_activate(self)
                        yield
                    finally:
                        self.cmdline.cache_deactivate(self)
                        self.running_time.cache_deactivate(self)

    def as_snapshot(
        self, attrs: Optional[Iterable[str]] = None, ad_value: Optional[Any] = None
    ) -> Snapshot:
        """Returns a onetime snapshot of the process."""

        with self.oneshot():
            attributes = self.as_dict(attrs=attrs, ad_value=ad_value)

            if attrs is None:
                for attr in ('command', 'running_time', 'running_time_human'):
                    try:
                        attributes[attr] = getattr(self, attr)()
                    except (host.AccessDenied, host.ZombieProcess):
                        attributes[attr] = ad_value

        return Snapshot(real=self, **attributes)


@HostProcess.register
class GpuProcess:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Represents a process with the given PID running on the given GPU device.
    The instance will be cache during the lifetime of the process.

    The same host process can use multiple GPU devices. The :class:`GpuProcess` instances representing the
    same PID on the host but different GPU devices are different.
    """

    INSTANCE_LOCK = threading.RLock()
    INSTANCES = WeakValueDictionary()

    # pylint: disable-next=too-many-arguments,unused-argument
    def __new__(
        cls,
        pid: int,
        device: 'Device',
        gpu_memory: Optional[Union[int, NaType]] = None,
        gpu_instance_id: Optional[Union[int, NaType]] = None,
        compute_instance_id: Optional[Union[int, NaType]] = None,
        type: Optional[Union[str, NaType]] = None,  # pylint: disable=redefined-builtin
    ) -> 'GpuProcess':
        """Returns the cached instance of :class:`GpuProcess`."""

        if pid is None:
            pid = os.getpid()

        with cls.INSTANCE_LOCK:
            try:
                instance = cls.INSTANCES[(pid, device)]
                if instance.is_running():
                    return instance
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

    # pylint: disable-next=too-many-arguments,unused-argument
    def __init__(
        self,
        pid: int,
        device: 'Device',
        gpu_memory: Optional[Union[int, NaType]] = None,
        gpu_instance_id: Optional[Union[int, NaType]] = None,
        compute_instance_id: Optional[Union[int, NaType]] = None,
        type: Optional[Union[str, NaType]] = None,  # pylint: disable=redefined-builtin
    ) -> None:
        """Initializes the instance returned by ``__new__()``."""

        if gpu_memory is None and not hasattr(self, '_gpu_memory'):
            gpu_memory = NA
        if gpu_memory is not None:
            self.set_gpu_memory(gpu_memory)

        if type is None and not hasattr(self, '_type'):
            type = NA
        if type is not None:
            self.type = type

        if gpu_instance_id is not None and compute_instance_id is not None:
            self._gpu_instance_id = gpu_instance_id if gpu_instance_id != 0xFFFFFFFF else NA
            self._compute_instance_id = (
                compute_instance_id if compute_instance_id != 0xFFFFFFFF else NA
            )
        elif device.is_mig_device():
            self._gpu_instance_id = device.gpu_instance_id()
            self._compute_instance_id = device.compute_instance_id()
        else:
            self._gpu_instance_id = self._compute_instance_id = NA

        for util in ('sm', 'memory', 'encoder', 'decoder'):
            if not hasattr(self, '_gpu_{}_utilization'.format(util)):
                setattr(self, '_gpu_{}_utilization'.format(util), NA)

    def __str__(self) -> str:
        return '{}(pid={}, gpu_memory={}, type={}, device={}, host={})'.format(
            self.__class__.__name__,
            self.pid,
            self.gpu_memory_human(),
            self.type,
            self.device,
            self.host,
        )

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (GpuProcess, host.Process)):
            return NotImplemented
        return self._ident == other._ident

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        if self._hash is None:  # pylint: disable=access-member-before-definition
            self._hash = hash(self._ident)  # pylint: disable=attribute-defined-outside-init
        return self._hash

    def __getattr__(self, name: str) -> Union[Any, Callable[..., Any]]:
        """Gets a member from the instance. Fallback to the host process instance if missing.

        Raises:
            AttributeError:
                If the attribute is not defined in either :class:`GpuProcess` nor :class:`HostProcess`.
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.
        """

        try:
            return super().__getattr__(name)
        except AttributeError:
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
    def device(self) -> 'Device':
        """The GPU device the process running on.

        The same host process can use multiple GPU devices.
        The :class:`GpuProcess` instances representing the same PID on the host
        but different GPU devices are different.
        """

        return self._device

    def gpu_instance_id(self) -> Union[int, NaType]:
        """The GPU instance ID of the MIG device, or :const:`nvitop.NA` if not applicable."""

        return self._gpu_instance_id

    def compute_instance_id(self) -> Union[int, NaType]:
        """The compute instance ID of the MIG device, or :const:`nvitop.NA` if not applicable."""

        return self._compute_instance_id

    def gpu_memory(self) -> Union[int, NaType]:  # in bytes
        """The used GPU memory in bytes, or :const:`nvitop.NA` if not applicable."""

        return self._gpu_memory

    def gpu_memory_human(self) -> Union[str, NaType]:  # in human readable
        """The used GPU memory in human readable format, or :const:`nvitop.NA` if not applicable."""

        return self._gpu_memory_human

    def gpu_memory_percent(self) -> Union[float, NaType]:  # in percentage
        """The percentage of used GPU memory by the process, or :const:`nvitop.NA` if not applicable."""

        return self._gpu_memory_percent

    def gpu_sm_utilization(self) -> Union[int, NaType]:  # in percentage
        """The utilization rate of SM (Streaming Multiprocessor), or :const:`nvitop.NA` if not applicable."""

        return self._gpu_sm_utilization

    def gpu_memory_utilization(self) -> Union[int, NaType]:  # in percentage
        """The utilization rate of GPU memory bandwidth, or :const:`nvitop.NA` if not applicable."""

        return self._gpu_memory_utilization

    def gpu_encoder_utilization(self) -> Union[int, NaType]:  # in percentage
        """The utilization rate of the encoder, or :const:`nvitop.NA` if not applicable."""

        return self._gpu_encoder_utilization

    def gpu_decoder_utilization(self) -> Union[int, NaType]:  # in percentage
        """The utilization rate of the decoder, or :const:`nvitop.NA` if not applicable."""

        return self._gpu_decoder_utilization

    def set_gpu_memory(self, value: Union[int, NaType]) -> None:
        """Sets the used GPU memory in bytes."""

        # pylint: disable=attribute-defined-outside-init
        self._gpu_memory = memory_used = value
        self._gpu_memory_human = bytes2human(self.gpu_memory())
        memory_total = self.device.memory_total()
        gpu_memory_percent = NA
        if libnvml.nvmlCheckReturn(memory_used, int) and libnvml.nvmlCheckReturn(memory_total, int):
            gpu_memory_percent = round(100.0 * memory_used / memory_total, 1)
        self._gpu_memory_percent = gpu_memory_percent

    def set_gpu_utilization(
        self,
        gpu_sm_utilization: Optional[int] = None,
        gpu_memory_utilization: Optional[int] = None,
        gpu_encoder_utilization: Optional[int] = None,
        gpu_decoder_utilization: Optional[int] = None,
    ) -> None:
        """Sets the GPU utilization rates."""

        # pylint: disable=attribute-defined-outside-init
        if gpu_sm_utilization is not None:
            self._gpu_sm_utilization = gpu_sm_utilization
        if gpu_memory_utilization is not None:
            self._gpu_memory_utilization = gpu_memory_utilization
        if gpu_encoder_utilization is not None:
            self._gpu_encoder_utilization = gpu_encoder_utilization
        if gpu_decoder_utilization is not None:
            self._gpu_decoder_utilization = gpu_decoder_utilization

    def update_gpu_status(self) -> Union[int, NaType]:
        """Updates the GPU consumption status from a new NVML query."""

        self.device.processes.cache_clear()
        self.device.processes()
        return self.gpu_memory()

    @property
    def type(self) -> Union[str, NaType]:
        """The type of the GPU context.

        The type is one of the following:
            - :data:`'C'`: compute context
            - :data:`'G'`: graphics context
            - :data:`'C+G'`: both compute context and graphics context
            - :data:`'N/A'`: not applicable
        """

        return self._type

    @type.setter
    def type(self, value: Union[str, NaType]) -> None:
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
        """Returns whether this process is running."""

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
    def create_time(self) -> Union[float, NaType]:
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
    def running_time(self) -> Union[datetime.timedelta, NaType]:
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

    def running_time_human(self) -> Union[str, NaType]:
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

    def running_time_in_seconds(self) -> Union[float, NaType]:
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
    def username(self) -> Union[str, NaType]:
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
    def name(self) -> Union[str, NaType]:
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
    def cpu_percent(self) -> Union[float, NaType]:  # in percentage
        """Returns a float representing the current process CPU utilization as a percentage.

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
    def memory_percent(self) -> Union[float, NaType]:  # in percentage
        """Compares process RSS memory to total physical system memory
        and calculate process memory utilization as a percentage.

        Raises:
            host.NoSuchProcess:
                If the process is gone.
            host.AccessDenied:
                If the user do not have read privilege to the process' status file.

        Note:
            To return the fallback value rather than raise an exception, please use the context
            manager :meth:`GpuProcess.failsafe`. See also :meth:`take_snapshots` and :meth:`failsafe`.
        """

        return self.host.memory_percent()

    host_memory_percent = memory_percent  # in percentage

    @auto_garbage_clean(fallback=NA)
    def host_memory(self) -> Union[int, NaType]:  # in bytes
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

    def host_memory_human(self) -> Union[str, NaType]:
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
    def cmdline(self) -> List[str]:
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
        """Returns a shell-escaped string from command line arguments.

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
        """Returns a onetime snapshot of the host process."""

        with self.host.oneshot():
            host_snapshot = Snapshot(
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

        return host_snapshot

    @auto_garbage_clean(fallback=_RAISE)
    def as_snapshot(
        self, *, host_process_snapshot_cache: Optional[Dict[int, Snapshot]] = None
    ) -> Snapshot:
        """Returns a onetime snapshot of the process on the GPU device.
        See also :meth:`take_snapshots` and :meth:`failsafe`.
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
        cls, gpu_processes: Iterable['GpuProcess'], *, failsafe=False
    ) -> List[Snapshot]:
        """Takes snapshots for a list of :class:`GpuProcess` instances.

        If *failsafe* is :data:`True`, then if any method fails, the fallback value in
        :func:`auto_garbage_clean` will be used.
        """

        cache = {}
        context = cls.failsafe if failsafe else contextlib.nullcontext
        with context():
            snapshots = [
                process.as_snapshot(host_process_snapshot_cache=cache) for process in gpu_processes
            ]

        return snapshots

    @classmethod
    @contextlib.contextmanager
    def failsafe(cls):
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
            ...     print('fallback (int cast):   {!r}'.format(float(p.cpu_percent())))  # `nvitop.NA` can be cast to float or int
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
