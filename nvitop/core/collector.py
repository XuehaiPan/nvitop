# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring

import contextlib
import itertools
import math
import os
import threading
import time
from collections import OrderedDict, defaultdict
from weakref import WeakSet
from typing import List, Dict, Iterable, NamedTuple, Hashable, Union, Optional

from nvitop.core import host
from nvitop.core.device import Device, CudaDevice
from nvitop.core.process import HostProcess, GpuProcess
from nvitop.core.utils import Snapshot, MiB, GiB


__all__ = ['take_snapshots', 'ResourceMetricCollector']


SnapshotResult = NamedTuple('SnapshotResult',  # in bytes
                            [('devices', List[Snapshot]),
                             ('gpu_processes', List[Snapshot])])


timer = time.monotonic


def take_snapshots(
    devices: Optional[Union[Device, Iterable[Device]]] = None, *,
    gpu_processes: Optional[Union[GpuProcess, Iterable[GpuProcess]]] = None
) -> SnapshotResult:
    """Retrieve status of demanded devices and GPU processes.

    Args:
        devices (Optional[Union[Device, Iterable[Device]]]):
            Requested devices for snapshots. If not given, the devices will be
            determined from GPU processes:
                - All devices (no GPU processes are given)
                - Devices that used by given GPU processes
        gpu_processes (Optional[Union[GpuProcess, Iterable[GpuProcess]]]):
            Requested GPU processes snapshots. If not given, all GPU processes
            running on the requested device will be returned.

    Returns:
        SnapshotResult: a named tuple containing two lists of snapshots

    Note:
        If not arguments are specified, all devices and all GPU processes will
        be returned.

    Examples:

        >>> from nvitop import take_snapshots, Device, CudaDevice
        >>> import os
        >>> os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

        >>> take_snapshots()  # equivalent to `take_snapshots(Device.all())`
        SnapshotResult(
            devices=[
                PhysicalDeviceSnapshot(
                    real=PhysicalDevice(index=0, ...),
                    ...
                ),
                ...
            ],
            gpu_processes=[
                GpuProcessSnapshot(
                    real=GpuProcess(pid=xxxxxx, device=PhysicalDevice(index=0, ...), ...),
                    ...
                ),
                ...
            ]
        )

        >>> take_snapshots(CudaDevice.all())
        SnapshotResult(
            devices=[
                CudaDeviceSnapshot(
                    real=CudaDevice(cuda_index=0, physical_index=1, ...),
                    ...
                ),
                CudaDeviceSnapshot(
                    real=CudaDevice(cuda_index=1, physical_index=0, ...),
                    ...
                ),
            ],
            gpu_processes=[
                GpuProcessSnapshot(
                    real=GpuProcess(pid=xxxxxx, device=CudaDevice(cuda_index=0, ...), ...),
                    ...
                ),
                ...
            ]
        )

        >>> take_snapshots(CudaDevice(1))  # <CUDA 1> only
        SnapshotResult(
            devices=[
                CudaDeviceSnapshot(
                    real=CudaDevice(cuda_index=1, physical_index=0, ...),
                    ...
                )
            ],
            gpu_processes=[
                GpuProcessSnapshot(
                    real=GpuProcess(pid=xxxxxx, device=CudaDevice(cuda_index=1, ...), ...),
                    ...
                ),
                ...
            ]
        )
    """

    def unique(iterable: Iterable[Hashable]) -> List[Hashable]:
        return list(OrderedDict.fromkeys(iterable).keys())

    if isinstance(devices, Device):
        devices = [devices]
    if isinstance(gpu_processes, GpuProcess):
        gpu_processes = [gpu_processes]

    if gpu_processes is not None:
        gpu_processes = list(gpu_processes)
        process_devices = unique(process.device for process in gpu_processes)
        for device in process_devices:
            device.processes()  # update GPU status for requested GPU processes
        if devices is None:
            devices = process_devices
    else:
        if devices is None:
            physical_devices = Device.all()
            devices = []
            leaf_devices = []
            for physical_device in physical_devices:
                devices.append(physical_device)
                mig_devices = physical_device.mig_devices()
                if len(mig_devices) > 0:
                    devices.extend(mig_devices)
                    leaf_devices.extend(mig_devices)
                else:
                    leaf_devices.append(physical_device)
        else:
            leaf_devices = devices = list(devices)
        gpu_processes = list(itertools.chain.from_iterable(device.processes().values()
                                                           for device in leaf_devices))

    devices = [device.as_snapshot() for device in devices]
    gpu_processes = GpuProcess.take_snapshots(gpu_processes, failsafe=True)

    return SnapshotResult(devices, gpu_processes)


class ResourceMetricCollector:  # pylint: disable=too-many-instance-attributes
    """A class for collecting resource metrics.

    Args:
        devices (iterable of Device):
            Set of Device instances for logging. If not given, all physical
            devices on board will be used.
        root_pids (set of int):
            A set of PIDs, only the status of the children processes on the GPUs
            will be collected. If not given, the PID of the current process will
            be used.
        interval (float): The interval between two snapshots.

    Core methods:

        collector.start(tag='<tag>')
        collector.stop()
        collector.collect()

        with collector(tag='<tag>'):
            ...

    Examples:

        >>> import os
        >>> os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'

        >>> from nvitop import ResourceMetricCollector, Device, CudaDevice

        >>> collector = ResourceMetricCollector()                          # log all devices and children processes on the GPUs of the current process
        >>> collector = ResourceMetricCollector(root_pids={1})             # log all devices and all GPU processes
        >>> collector = ResourceMetricCollector(devices=CudaDevice.all())  # use the CUDA ordinal

        >>> with collector(tag='<tag>'):
        ...     # do something
        ...     collector.collect()  # -> Dict[str, float]
        {
            '<tag>/host/cpu_percent (%)': 8.967849777683456,
            '<tag>/host/memory_percent (%)': 21.5,
            '<tag>/host/swap_percent (%)': 0.3,
            '<tag>/host/memory_used (GiB)': 91.0136418208109,
            '<tag>/host/load_average (%) (1 min)': 10.251427386878328,
            '<tag>/host/load_average (%) (5 min)': 10.072539414569503,
            '<tag>/host/load_average (%) (15 min)': 11.91126970422139,
            '<tag>/cuda:0 (gpu:3)/memory_used (MiB)': 3.875,
            '<tag>/cuda:0 (gpu:3)/memory_free (MiB)': 11015.562499999998,
            '<tag>/cuda:0 (gpu:3)/memory_total (MiB)': 11019.437500000002,
            '<tag>/cuda:0 (gpu:3)/memory_percent (%)': 0.0,
            '<tag>/cuda:0 (gpu:3)/gpu_utilization (%)': 0.0,
            '<tag>/cuda:0 (gpu:3)/memory_utilization (%)': 0.0,
            '<tag>/cuda:0 (gpu:3)/fan_speed (%)': 22.0,
            '<tag>/cuda:0 (gpu:3)/temperature (℃)': 25.0,
            '<tag>/cuda:0 (gpu:3)/power_usage (W)': 19.11166264116916,
            '<tag>/cuda:1 (gpu:2)/memory_used (MiB)': 8878.875,
            ...,
            '<tag>/cuda:2 (gpu:1)/memory_used (MiB)': 8182.875,
            ...,
            '<tag>/cuda:3 (gpu:0)/memory_used (MiB)': 9286.875,
            ...,
            '<tag>/pid:12345/cuda:1 (gpu:4)/cpu_percent (%)': 151.34342772112265,
            '<tag>/pid:12345/cuda:1 (gpu:4)/host_memory (MiB)': 44749.72373447514,
            '<tag>/pid:12345/cuda:1 (gpu:4)/host_memory_percent (%)': 8.675082352111717,
            '<tag>/pid:12345/cuda:1 (gpu:4)/running_time (min)': 336.23803206741576,
            '<tag>/pid:12345/cuda:1 (gpu:4)/gpu_memory (MiB)': 8861.0,
            '<tag>/pid:12345/cuda:1 (gpu:4)/gpu_memory_percent (%)': 80.4,
            '<tag>/pid:12345/cuda:1 (gpu:4)/gpu_memory_utilization (%)': 6.711118172407917,
            '<tag>/pid:12345/cuda:1 (gpu:4)/gpu_sm_utilization (%)': 48.23283397736476,
            ...,
            '<tag>/duration (s)': 7.247399162035435,
            '<tag>/timestamp': 1655909466.9981883
        }
    """  # pylint: disable=line-too-long

    DEVICE_METRICS = [
        # (<attribute>, <name>, <unit>)
        # GPU memory metrics
        ('memory_used', 'memory_used (MiB)', MiB),
        ('memory_free', 'memory_free (MiB)', MiB),
        ('memory_total', 'memory_total (MiB)', MiB),
        ('memory_percent', 'memory_percent (%)', 1.0),

        # GPU utilization metrics
        ('gpu_utilization', 'gpu_utilization (%)', 1.0),
        ('memory_utilization', 'memory_utilization (%)', 1.0),

        # Miscellaneous
        ('fan_speed', 'fan_speed (%)', 1.0),
        ('temperature', 'temperature (℃)', 1.0),
        ('power_usage', 'power_usage (W)', 1000.0),
    ]

    PROCESS_METRICS = [
        # (<attribute>, <name>, <unit>)
        # Host resource metrics
        ('cpu_percent', 'cpu_percent (%)', 1.0),
        ('host_memory', 'host_memory (MiB)', MiB),
        ('host_memory_percent', 'host_memory_percent (%)', 1.0),
        ('running_time_in_seconds', 'running_time (min)', 60.0),

        # GPU memory metrics
        ('gpu_memory', 'gpu_memory (MiB)', MiB),
        ('gpu_memory_percent', 'gpu_memory_percent (%)', 1.0),
        ('gpu_memory_utilization', 'gpu_memory_utilization (%)', 1.0),

        # GPU utilization metrics
        ('gpu_sm_utilization', 'gpu_sm_utilization (%)', 1.0),
    ]

    def __init__(
        self,
        devices: Optional[Iterable[Device]] = None,
        root_pids: Optional[Iterable[int]] = None,
        interval: Union[int, float] = 1.0
    ) -> None:
        if isinstance(interval, (int, float)) and interval > 0:
            interval = float(interval)
        else:
            raise ValueError('Invalid argument interval={:!r}'.format(interval))

        if devices is None:
            devices = Device.all()

        if root_pids is None:
            root_pids = {os.getpid()}
        else:
            root_pids = set(root_pids)

        self.interval = interval

        self.devices = list(devices)
        self.all_devices = []
        self.leaf_devices = []
        for device in self.devices:
            self.all_devices.append(device)
            mig_devices = device.mig_devices()
            if len(mig_devices) > 0:
                self.all_devices.extend(mig_devices)
                self.leaf_devices.extend(mig_devices)
            else:
                self.leaf_devices.append(device)

        self.root_pids = root_pids
        self._positive_processes = WeakSet(HostProcess(pid) for pid in self.root_pids)
        self._negative_processes = WeakSet()

        self._last_timestamp = timer() - 2.0 * self.interval
        self._lock = threading.RLock()
        self._metric_buffer = None
        self._tags = set()

        self._daemon = threading.Thread(name='gpu_metric_collector_daemon', target=self._target, daemon=True)
        self._daemon_running = threading.Event()

    def start(self, tag: str) -> None:
        """Start a new metric collection with the given tag.

        Args:
            tag (str):
                The name of the new metric collection. The tag will be used to
                identify the metric collection. It must be a unique string.

        Examples:

            >>> collector = ResourceMetricCollector()

            >>> collector.start(tag='train')  # tag -> 'train'
            >>> collector.start(tag='batch')  # tag -> 'train/batch'
            >>> collector.stop()              # tag -> 'train'
            >>> collector.stop()              # the collector has been stopped
            >>> collector.start(tag='test')   # tag -> 'test'
        """

        with self._lock:
            if self._metric_buffer is None or tag not in self._tags:
                self._tags.add(tag)
                self._metric_buffer = _MetricBuffer(tag, self, prev=self._metric_buffer)
                self._last_timestamp = timer() - 2.0 * self.interval
            else:
                raise RuntimeError(
                    'Resource metric collector is already started with tag "{}"'.format(tag)
                )

        self._daemon_running.set()
        try:
            self._daemon.start()
        except RuntimeError:
            pass

    def stop(self) -> None:
        """Stop the current collection and pop out the last nested tag."""

        with self._lock:
            if self._metric_buffer is not None:
                self._tags.remove(self._metric_buffer.tag)
                self._metric_buffer = self._metric_buffer.prev
            else:
                self._daemon_running.clear()

    @contextlib.contextmanager
    def context(self, tag: str) -> 'ResourceMetricCollector':
        """A context manager for starting and stopping resource metric collection.

        Args:
            tag (str):
                The name of the new metric collection. The tag will be used to
                identify the metric collection. It must be a unique string.

        Examples:

            >>> collector = ResourceMetricCollector()

            >>> with collector(tag='train'):  # tag -> 'train'
            ...     # do something
            ...     collector.collect()  # -> Dict[str, float]
        """

        try:
            self.start(tag=tag)
            yield self
        finally:
            self.stop()

    __call__ = context

    def collect(self) -> Dict[str, float]:
        """Get the average resource consumption during collection."""

        with self._lock:
            if self._metric_buffer is None:
                raise RuntimeError(
                    'Resource metric collector has not been not started yet.'
                )

            if timer() - self._last_timestamp > self.interval:
                self.take_snapshots()
            return self._metric_buffer.collect()

    def __del__(self) -> None:
        self.stop()

    def take_snapshots(self) -> SnapshotResult:  # pylint: disable=missing-function-docstring,too-many-branches,too-many-locals
        if len(self.root_pids) > 0:
            all_gpu_processes = []
            for device in self.leaf_devices:
                all_gpu_processes.extend(device.processes().values())

            gpu_processes = []
            for process in all_gpu_processes:
                if process.host in self._negative_processes:
                    continue

                positive = True
                if process.host not in self._positive_processes:
                    positive = False
                    p = process.host  # pylint: disable=invalid-name
                    parents = []
                    while p is not None:
                        parents.append(p)
                        if p in self._positive_processes:
                            positive = True
                            break
                        try:
                            p = p.parent()  # pylint: disable=invalid-name
                        except host.PsutilError:
                            break
                    if positive:
                        self._positive_processes.update(parents)
                    else:
                        self._negative_processes.update(parents)

                if positive:
                    gpu_processes.append(process)
        else:
            gpu_processes = []

        timestamp = timer()
        metrics = {}
        devices = [device.as_snapshot() for device in self.all_devices]
        gpu_processes = GpuProcess.take_snapshots(gpu_processes, failsafe=True)

        metrics.update({
            'host/cpu_percent (%)': host.cpu_percent(),
            'host/memory_percent (%)': host.memory_percent(),
            'host/swap_percent (%)': host.swap_percent(),
            'host/memory_used (GiB)': host.virtual_memory().used / GiB,
        })
        load_average = host.load_average()
        if load_average is not None:
            metrics.update({
                'host/load_average (%) (1 min)': load_average[0],
                'host/load_average (%) (5 min)': load_average[1],
                'host/load_average (%) (15 min)': load_average[2],
            })

        device_identifiers = {}
        for device in devices:
            identifier = 'gpu:{}'.format(device.index)
            if isinstance(device.real, CudaDevice):
                identifier = 'cuda:{} ({})'.format(device.cuda_index, identifier)
            device_identifiers[device.real] = identifier

            for attr, name, unit in self.DEVICE_METRICS:
                value = float(getattr(device, attr)) / unit
                metrics['{}/{}'.format(identifier, name)] = value

        for process in gpu_processes:
            identifier = 'pid:{}/{}'.format(process.pid, device_identifiers[process.device])

            for attr, name, unit in self.PROCESS_METRICS:
                value = float(getattr(process, attr)) / unit
                metrics['{}/{}'.format(identifier, name)] = value

        with self._lock:
            if self._metric_buffer is not None:
                self._metric_buffer.add(metrics, timestamp=timestamp)
                self._last_timestamp = timestamp

        return SnapshotResult(devices, gpu_processes)

    def _target(self) -> None:
        self._daemon_running.wait()
        while self._daemon_running.is_set():
            self.take_snapshots()
            time.sleep(self.interval)


class _MetricBuffer:  # pylint: disable=missing-class-docstring,missing-function-docstring,too-many-instance-attributes
    def __init__(self, tag: str,
                 collector: 'ResourceMetricCollector',
                 prev: Optional['_MetricBuffer'] = None) -> None:
        self.collector = collector
        self.prev = prev

        self.tag = tag
        if self.prev is not None:
            self.key_prefix = '{}/{}'.format(self.prev.tag, self.tag)
        else:
            self.key_prefix = self.tag

        self.last_timestamp = self.start_timestamp = timer()
        self.buffer = defaultdict(lambda: _MeanValueMaintainer(self.last_timestamp))

        self.len = 0

    def add(self, metrics: Dict[str, float], timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = timer()

        for key, value in metrics.items():
            self.buffer[key].add(value, timestamp=timestamp)
        self.len += 1
        self.last_timestamp = timestamp

        if self.prev is not None:
            self.prev.add(metrics, timestamp=timestamp)

    def collect(self) -> Dict[str, float]:
        metrics = {'{}/{}'.format(self.key_prefix, key): maintainer.mean()
                   for key, maintainer in self.buffer.items()}
        metrics['{}/duration (s)'.format(self.key_prefix)] = timer() - self.start_timestamp
        metrics['{}/timestamp'.format(self.key_prefix)] = time.time()
        return metrics

    def __len__(self) -> int:
        return self.len


class _MeanValueMaintainer:  # pylint: disable=missing-class-docstring,missing-function-docstring
    def __init__(self, timestamp: float) -> None:
        self.start_timestamp = timestamp
        self.integral = None
        self.last_value = None
        self.last_timestamp = None
        self.has_nan = False

    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = timer()

        if math.isnan(value):
            self.has_nan = True
            return

        if self.last_value is None:
            self.integral = value * (timestamp - self.start_timestamp)
            self.last_value = value
        else:
            self.integral += (value + self.last_value) * (timestamp - self.last_timestamp) / 2.0
            self.last_value = value

        self.last_timestamp = timestamp

    def mean(self) -> float:
        if self.has_nan:
            if self.integral is None:
                return math.nan
            return self.integral / (self.last_timestamp - self.start_timestamp)

        timestamp = timer()
        integral = self.integral + self.last_value * (timestamp - self.last_timestamp)
        return integral / (timestamp - self.start_timestamp)
