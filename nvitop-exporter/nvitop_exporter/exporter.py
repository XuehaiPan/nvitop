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
"""Prometheus exporter built on top of ``nvitop``."""

from __future__ import annotations

import math
import time
from typing import Sequence

from prometheus_client import REGISTRY, CollectorRegistry, Gauge, Info

from nvitop import Device, MiB, MigDevice, PhysicalDevice, host
from nvitop.api.process import GpuProcess
from nvitop_exporter.utils import get_ip_address


class PrometheusExporter:  # pylint: disable=too-many-instance-attributes
    """Prometheus exporter built on top of ``nvitop``."""

    def __init__(  # pylint: disable=too-many-statements
        self,
        devices: Sequence[Device],
        hostname: str | None = None,
        *,
        registry: CollectorRegistry = REGISTRY,
        interval: float = 1.0,
    ) -> None:
        """Initialize the Prometheus exporter."""
        if not isinstance(devices, (list, tuple)):
            raise TypeError(f'Expected a list or tuple of devices, got {type(devices)}')
        devices = list(devices)

        for device in devices:
            if not isinstance(device, (PhysicalDevice, MigDevice)):
                raise TypeError(f'Expected a PhysicalDevice or MigDevice, got {type(device)}')

        self.devices = devices
        self.hostname = hostname or get_ip_address()
        self.registry = registry
        self.interval = interval
        self.alive_pids: dict[Device, set[tuple[int, str]]] = {
            device: set() for device in self.devices
        }

        self.info = Info(
            'nvitop',
            documentation='NVITOP Prometheus Exporter.',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.info.labels(hostname=self.hostname).info(
            {
                'device_count': str(Device.count()),
                'driver_version': Device.driver_version(),
                'cuda_driver_version': Device.cuda_driver_version(),
            },
        )

        # Create gauges for host metrics
        self.host_uptime = Gauge(
            name='host_uptime',
            documentation='Host uptime (s).',
            unit='Second',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_cpu_percent = Gauge(
            name='host_cpu_percent',
            documentation='Host CPU percent (%).',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_virtual_memory_total = Gauge(
            name='host_virtual_memory_total',
            documentation='Host virtual memory total (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_virtual_memory_used = Gauge(
            name='host_virtual_memory_used',
            documentation='Host virtual memory used (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_virtual_memory_free = Gauge(
            name='host_virtual_memory_free',
            documentation='Host virtual memory free (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_virtual_memory_percent = Gauge(
            name='host_virtual_memory_percent',
            documentation='Host virtual memory percent (%).',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_swap_memory_total = Gauge(
            name='host_swap_memory_total',
            documentation='Host swap total (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_swap_memory_used = Gauge(
            name='host_swap_memory_used',
            documentation='Host swap used (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_swap_memory_free = Gauge(
            name='host_swap_memory_free',
            documentation='Host swap free (MiB).',
            unit='MiB',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_swap_memory_percent = Gauge(
            name='host_swap_memory_percent',
            documentation='Host swap percent (%).',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_load_average_1m = Gauge(
            name='host_load_average_1m',
            documentation='Host load average for the last minute.',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_load_average_5m = Gauge(
            name='host_load_average_5m',
            documentation='Host load average for the last 5 minutes.',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_load_average_15m = Gauge(
            name='host_load_average_15m',
            documentation='Host load average for the last 15 minutes.',
            unit='Percentage',
            labelnames=['hostname'],
            registry=self.registry,
        )
        self.host_net_io_tx_data = Gauge(
            name='host_net_io_tx_data',
            documentation='Host network I/O transmitted data (MiB).',
            unit='MiB',
            labelnames=['hostname', 'interface'],
            registry=self.registry,
        )
        self.host_net_io_rx_data = Gauge(
            name='host_net_io_rx_data',
            documentation='Host network I/O received data (MiB).',
            unit='MiB',
            labelnames=['hostname', 'interface'],
            registry=self.registry,
        )
        self.host_net_io_tx_packets = Gauge(
            name='host_net_io_tx_packets',
            documentation='Host network I/O transmitted packets.',
            unit='Packet',
            labelnames=['hostname', 'interface'],
            registry=self.registry,
        )
        self.host_net_io_rx_packets = Gauge(
            name='host_net_io_rx_packets',
            documentation='Host network I/O received packets.',
            unit='Packet',
            labelnames=['hostname', 'interface'],
            registry=self.registry,
        )
        self.host_disk_io_read_data = Gauge(
            name='host_disk_io_read_data',
            documentation='Host disk I/O read data (MiB).',
            unit='MiB',
            labelnames=['hostname', 'partition'],
            registry=self.registry,
        )
        self.host_disk_io_write_data = Gauge(
            name='host_disk_io_write_data',
            documentation='Host disk I/O write data (MiB).',
            unit='MiB',
            labelnames=['hostname', 'partition'],
            registry=self.registry,
        )
        self.host_disk_usage_total = Gauge(
            name='host_disk_usage_total',
            documentation='Host disk usage total (MiB).',
            unit='MiB',
            labelnames=['hostname', 'mountpoint'],
            registry=self.registry,
        )
        self.host_disk_usage_used = Gauge(
            name='host_disk_usage_used',
            documentation='Host disk usage used (MiB).',
            unit='MiB',
            labelnames=['hostname', 'mountpoint'],
            registry=self.registry,
        )
        self.host_disk_usage_free = Gauge(
            name='host_disk_usage_free',
            documentation='Host disk usage free (MiB).',
            unit='MiB',
            labelnames=['hostname', 'mountpoint'],
            registry=self.registry,
        )
        self.host_disk_usage_percent = Gauge(
            name='host_disk_usage_percent',
            documentation='Host disk usage percent (%).',
            unit='Percentage',
            labelnames=['hostname', 'mountpoint'],
            registry=self.registry,
        )

        # Create gauges for GPU metrics
        self.gpu_utilization = Gauge(
            name='gpu_utilization',
            documentation='GPU utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_utilization = Gauge(
            name='gpu_memory_utilization',
            documentation='GPU memory utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_encoder_utilization = Gauge(
            name='gpu_encoder_utilization',
            documentation='GPU encoder utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_decoder_utilization = Gauge(
            name='gpu_decoder_utilization',
            documentation='GPU decoder utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_total = Gauge(
            name='gpu_memory_total',
            documentation='GPU memory total (MiB).',
            unit='MiB',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_used = Gauge(
            name='gpu_memory_used',
            documentation='GPU memory used (MiB).',
            unit='MiB',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_free = Gauge(
            name='gpu_memory_free',
            documentation='GPU memory free (MiB).',
            unit='MiB',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_memory_percent = Gauge(
            name='gpu_memory_percent',
            documentation='GPU memory percent (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_sm = Gauge(
            name='gpu_clock_sm',
            documentation='GPU SM clock (MHz).',
            unit='MHz',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_memory = Gauge(
            name='gpu_clock_memory',
            documentation='GPU memory clock (MHz).',
            unit='MHz',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_graphics = Gauge(
            name='gpu_clock_graphics',
            documentation='GPU graphics clock (MHz).',
            unit='MHz',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_clock_video = Gauge(
            name='gpu_clock_video',
            documentation='GPU video clock (MHz).',
            unit='MHz',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_power_usage = Gauge(
            name='gpu_power_usage',
            documentation='GPU power usage (W).',
            unit='W',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_power_limit = Gauge(
            name='gpu_power_limit',
            documentation='GPU power limit (W).',
            unit='W',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_temperature = Gauge(
            name='gpu_temperature',
            documentation='GPU temperature (C).',
            unit='C',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_fan_speed = Gauge(
            name='gpu_fan_speed',
            documentation='GPU fan speed (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_pcie_tx_throughput = Gauge(
            name='gpu_pcie_tx_throughput',
            documentation='GPU PCIe transmit throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_pcie_rx_throughput = Gauge(
            name='gpu_pcie_rx_throughput',
            documentation='GPU PCIe receive throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_total_tx_throughput = Gauge(
            name='gpu_nvlink_total_tx_throughput',
            documentation='GPU total NVLink transmit throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_total_rx_throughput = Gauge(
            name='gpu_nvlink_total_rx_throughput',
            documentation='GPU total NVLink receive throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_mean_tx_throughput = Gauge(
            name='gpu_nvlink_mean_tx_throughput',
            documentation='GPU mean NVLink transmit throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_mean_rx_throughput = Gauge(
            name='gpu_nvlink_mean_rx_throughput',
            documentation='GPU mean NVLink receive throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid'],
            registry=self.registry,
        )
        self.gpu_nvlink_tx_throughput = Gauge(
            name='gpu_nvlink_tx_throughput',
            documentation='GPU NVLink transmit throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'link'],
            registry=self.registry,
        )
        self.gpu_nvlink_rx_throughput = Gauge(
            name='gpu_nvlink_rx_throughput',
            documentation='GPU NVLink receive throughput (MiB/s).',
            unit='MiBps',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'link'],
            registry=self.registry,
        )

        # Create gauges for process metrics
        self.process_info = Info(
            name='process_info',
            documentation='Process information.',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_running_time = Gauge(
            name='process_running_time',
            documentation='Process running time (s).',
            unit='Second',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_cpu_percent = Gauge(
            name='process_cpu_percent',
            documentation='Process CPU percent (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_rss_memory = Gauge(
            name='process_rss_memory',
            documentation='Process memory resident set size (MiB).',
            unit='MiB',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_memory_percent = Gauge(
            name='process_memory_percent',
            documentation='Process memory percent (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_gpu_memory = Gauge(
            name='process_gpu_memory',
            documentation='Process GPU memory (MiB).',
            unit='MiB',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_gpu_sm_utilization = Gauge(
            name='process_gpu_sm_utilization',
            documentation='Process GPU SM utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_gpu_memory_utilization = Gauge(
            name='process_gpu_memory_utilization',
            documentation='Process GPU memory utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_gpu_encoder_utilization = Gauge(
            name='process_gpu_encoder_utilization',
            documentation='Process GPU encoder utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )
        self.process_gpu_decoder_utilization = Gauge(
            name='process_gpu_decoder_utilization',
            documentation='Process GPU decoder utilization (%).',
            unit='Percentage',
            labelnames=['hostname', 'index', 'devicename', 'uuid', 'pid', 'username'],
            registry=self.registry,
        )

    def collect(self) -> None:
        """Collect metrics."""
        while True:
            next_update_time = time.monotonic() + self.interval
            self.update_host()
            for device in self.devices:
                self.update_device(device)
            time.sleep(max(0.0, next_update_time - time.monotonic()))

    def update_host(self) -> None:
        """Update metrics for the host."""
        load_average = host.load_average()
        if load_average is None:
            load_average = (0.0, 0.0, 0.0)  # type: ignore[unreachable]
        virtual_memory = host.virtual_memory()
        swap_memory = host.swap_memory()
        net_io_counters = host.net_io_counters(pernic=True)  # type: ignore[attr-defined]
        disk_io_counters = host.disk_io_counters(perdisk=True)  # type: ignore[attr-defined]

        for gauge, value in (
            (self.host_uptime, host.uptime()),
            (self.host_cpu_percent, host.cpu_percent()),
            (self.host_virtual_memory_total, virtual_memory.total / MiB),
            (self.host_virtual_memory_used, virtual_memory.used / MiB),
            (self.host_virtual_memory_free, virtual_memory.free / MiB),
            (self.host_virtual_memory_percent, virtual_memory.percent),
            (self.host_swap_memory_total, swap_memory.total / MiB),
            (self.host_swap_memory_used, swap_memory.used / MiB),
            (self.host_swap_memory_free, swap_memory.free / MiB),
            (self.host_swap_memory_percent, swap_memory.percent),
            (self.host_load_average_1m, load_average[0]),
            (self.host_load_average_5m, load_average[1]),
            (self.host_load_average_15m, load_average[2]),
        ):
            gauge.labels(self.hostname).set(value)

        for interface, net_io_counter in net_io_counters.items():
            for gauge, value in (
                (self.host_net_io_tx_data, net_io_counter.bytes_sent / MiB),
                (self.host_net_io_rx_data, net_io_counter.bytes_recv / MiB),
                (self.host_net_io_tx_packets, net_io_counter.packets_sent),
                (self.host_net_io_rx_packets, net_io_counter.packets_recv),
            ):
                gauge.labels(hostname=self.hostname, interface=interface).set(value)

        for partition, disk_io_counter in disk_io_counters.items():
            for gauge, value in (
                (self.host_disk_io_read_data, disk_io_counter.read_bytes / MiB),
                (self.host_disk_io_write_data, disk_io_counter.write_bytes / MiB),
            ):
                gauge.labels(hostname=self.hostname, partition=partition).set(value)

        for partition in host.disk_partitions():  # type: ignore[attr-defined]
            try:
                partition_usage = host.disk_usage(partition.mountpoint)  # type: ignore[attr-defined]
            except (OSError, host.PsutilError):
                continue
            for gauge, value in (
                (self.host_disk_usage_total, partition_usage.total / MiB),
                (self.host_disk_usage_used, partition_usage.used / MiB),
                (self.host_disk_usage_free, partition_usage.free / MiB),
                (self.host_disk_usage_percent, partition_usage.percent),
            ):
                gauge.labels(hostname=self.hostname, mountpoint=partition.mountpoint).set(value)

    def update_device(self, device: Device) -> None:  # pylint: disable=too-many-locals
        """Update metrics for a single device."""
        index = (
            str(device.index) if isinstance(device.index, int) else ':'.join(map(str, device.index))
        )
        name = device.name()
        uuid = device.uuid()

        with device.oneshot():
            for gauge, value in (
                (self.gpu_utilization, float(device.gpu_utilization())),
                (self.gpu_memory_utilization, float(device.memory_utilization())),
                (self.gpu_encoder_utilization, float(device.encoder_utilization())),
                (self.gpu_decoder_utilization, float(device.decoder_utilization())),
                (self.gpu_memory_total, device.memory_total() / MiB),
                (self.gpu_memory_used, device.memory_used() / MiB),
                (self.gpu_memory_free, device.memory_free() / MiB),
                (self.gpu_memory_percent, float(device.memory_percent())),
                (self.gpu_clock_sm, float(device.clock_infos().sm)),
                (self.gpu_clock_memory, float(device.clock_infos().memory)),
                (self.gpu_clock_graphics, float(device.clock_infos().graphics)),
                (self.gpu_clock_video, float(device.clock_infos().video)),
                (self.gpu_power_usage, device.power_usage() / 1000.0),
                (self.gpu_power_limit, device.power_limit() / 1000.0),
                (self.gpu_temperature, float(device.temperature())),
                (self.gpu_fan_speed, float(device.fan_speed())),
                (self.gpu_pcie_tx_throughput, device.pcie_tx_throughput() / 1024.0),
                (self.gpu_pcie_rx_throughput, device.pcie_rx_throughput() / 1024.0),
                (self.gpu_nvlink_total_tx_throughput, device.nvlink_total_tx_throughput() / 1024.0),
                (self.gpu_nvlink_total_rx_throughput, device.nvlink_total_rx_throughput() / 1024.0),
                (self.gpu_nvlink_mean_tx_throughput, device.nvlink_mean_tx_throughput() / 1024.0),
                (self.gpu_nvlink_mean_rx_throughput, device.nvlink_mean_rx_throughput() / 1024.0),
            ):
                gauge.labels(
                    hostname=self.hostname,
                    index=index,
                    devicename=name,
                    uuid=uuid,
                ).set(value)

            for gauge, nvlink_throughput in (
                (self.gpu_nvlink_tx_throughput, device.nvlink_tx_throughput()),
                (self.gpu_nvlink_rx_throughput, device.nvlink_rx_throughput()),
            ):
                for link, throughput in enumerate(nvlink_throughput):
                    gauge.labels(
                        hostname=self.hostname,
                        index=index,
                        devicename=name,
                        uuid=uuid,
                        link=link,
                    ).set(throughput / 1024.0)

        alive_pids = self.alive_pids[device]
        previous_alive_pids = alive_pids.copy()
        alive_pids.clear()

        with GpuProcess.failsafe():
            host_snapshots = {}
            for pid, process in device.processes().items():
                with process.oneshot():
                    username = process.username()
                    alive_pids.add((pid, username))
                    if (pid, username) not in host_snapshots:  # noqa: SIM401,RUF100
                        host_snapshot = host_snapshots[(pid, username)] = process.host_snapshot()
                    else:
                        host_snapshot = host_snapshots[(pid, username)]
                    self.process_info.labels(
                        hostname=self.hostname,
                        index=index,
                        devicename=name,
                        uuid=uuid,
                        pid=pid,
                        username=username,
                    ).info(
                        {
                            'status': host_snapshot.status,
                            'command': host_snapshot.command,
                        },
                    )
                    for gauge, value in (
                        (
                            self.process_running_time,
                            (
                                host_snapshot.running_time.total_seconds()
                                if host_snapshot.running_time
                                else math.nan
                            ),
                        ),
                        (self.process_cpu_percent, host_snapshot.cpu_percent),
                        (self.process_rss_memory, host_snapshot.host_memory / MiB),
                        (self.process_memory_percent, float(host_snapshot.memory_percent)),
                        (self.process_gpu_memory, process.gpu_memory() / MiB),
                        (
                            self.process_gpu_sm_utilization,
                            float(process.gpu_sm_utilization()),
                        ),
                        (
                            self.process_gpu_memory_utilization,
                            float(process.gpu_memory_utilization()),
                        ),
                        (
                            self.process_gpu_encoder_utilization,
                            float(process.gpu_encoder_utilization()),
                        ),
                        (
                            self.process_gpu_decoder_utilization,
                            float(process.gpu_decoder_utilization()),
                        ),
                    ):
                        gauge.labels(
                            hostname=self.hostname,
                            index=index,
                            devicename=name,
                            uuid=uuid,
                            pid=pid,
                            username=username,
                        ).set(value)

        for pid, username in previous_alive_pids.difference(alive_pids):
            for collector in (
                self.process_info,
                self.process_running_time,
                self.process_cpu_percent,
                self.process_rss_memory,
                self.process_memory_percent,
                self.process_gpu_memory,
                self.process_gpu_sm_utilization,
                self.process_gpu_memory_utilization,
                self.process_gpu_encoder_utilization,
                self.process_gpu_decoder_utilization,
            ):
                try:
                    collector.remove(
                        self.hostname,
                        index,
                        name,
                        uuid,
                        pid,
                        username,
                    )
                except KeyError:  # noqa: PERF203
                    pass
