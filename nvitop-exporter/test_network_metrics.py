"""Regression coverage for network metric life cycle, without GPU hardware."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry

from nvitop_exporter.exporter import PrometheusExporter


@pytest.fixture(name='exporter')
def isolated_exporter() -> Iterator[PrometheusExporter]:
    """Create an isolated collector without initializing GPU hardware."""
    mocked_values: tuple[tuple[str, object], ...] = (
        ('Device.count', 0),
        ('Device.driver_version', 'test'),
        ('Device.cuda_driver_version', 'test'),
        ('host.disk_partitions', []),
    )
    with ExitStack() as stack:
        for name, value in mocked_values:
            stack.enter_context(patch(f'nvitop_exporter.exporter.{name}', return_value=value))
        yield PrometheusExporter([], hostname='test', registry=CollectorRegistry())


def update(exporter: PrometheusExporter, *interfaces: str) -> None:
    """Collect a synthetic interface snapshot."""
    snapshot = {
        name: SimpleNamespace(
            bytes_sent=1048576,
            bytes_recv=2097152,
            packets_sent=3,
            packets_recv=4,
        )
        for name in interfaces
    }
    with patch('nvitop_exporter.exporter.host.net_io_counters', return_value=snapshot):
        exporter.update_host()


def assert_interfaces(exporter: PrometheusExporter, expected: set[str]) -> None:
    """Check all four network families and their values."""
    for gauge, value in (
        (exporter.host_net_io_tx_data, 1),
        (exporter.host_net_io_rx_data, 2),
        (exporter.host_net_io_tx_packets, 3),
        (exporter.host_net_io_rx_packets, 4),
    ):
        samples = gauge.collect()[0].samples
        assert {sample.labels['interface'] for sample in samples} == expected
        assert all(sample.labels['hostname'] == 'test' for sample in samples)
        assert all(sample.value == value for sample in samples)


def test_disappeared_interface_is_removed_and_can_return(exporter: PrometheusExporter) -> None:
    """Remove vanished labels while allowing recreation and empty snapshots."""
    update(exporter, 'eth0', 'veth0')
    assert_interfaces(exporter, {'eth0', 'veth0'})
    update(exporter, 'eth0')
    assert_interfaces(exporter, {'eth0'})
    update(exporter, 'eth0', 'veth0')
    assert_interfaces(exporter, {'eth0', 'veth0'})
    update(exporter)
    assert_interfaces(exporter, set())


def test_failed_snapshot_preserves_previous_metrics(exporter: PrometheusExporter) -> None:
    """Do not treat collection failure as an empty network."""
    update(exporter, 'eth0', 'veth0')
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                'nvitop_exporter.exporter.host.net_io_counters',
                side_effect=OSError('unavailable'),
            ),
        )
        stack.enter_context(pytest.raises(OSError, match='unavailable'))
        exporter.update_host()
    assert_interfaces(exporter, {'eth0', 'veth0'})
    update(exporter, 'eth0')
    assert_interfaces(exporter, {'eth0'})


def test_interface_churn_does_not_accumulate_metrics(exporter: PrometheusExporter) -> None:
    """Bound sample count to the interfaces present in the latest snapshot."""
    for index in range(200):
        update(exporter, 'eth0', f'veth{index}')
        assert_interfaces(exporter, {'eth0', f'veth{index}'})
