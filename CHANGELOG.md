# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [Unreleased]

### Added

- Add `nvidia-ml-py` 13.580.82 to support list for NVIDIA Spark/Thor by [@johnnynunez](https://github.com/johnnynunez) in [#186](https://github.com/XuehaiPan/nvitop/pull/186).
- Add bar charts for memory bandwidth and power usage in the main screen by [@XuehaiPan](https://github.com/XuehaiPan) in [#190](https://github.com/XuehaiPan/nvitop/pull/190).

### Changed

-

### Fixed

-

### Removed

-

------

## [1.5.3] - 2025-08-16

### Added

- Add CUDA-13 NVML API support by [@XuehaiPan](https://github.com/XuehaiPan) in [#178](https://github.com/XuehaiPan/nvitop/pull/178).

### Changed

- Draw network and disk I/O graphs with centered symmetric zero in Grafana dashboard by [@XuehaiPan](https://github.com/XuehaiPan).

### Fixed

- Fix `device.pcie_tx_throughput()` returns PCIe RX throughput due to a typo in argument by [@kyet](https://github.com/kyet) in [#176](https://github.com/XuehaiPan/nvitop/pull/176).

### Removed

- Remove per-version install extras for `nvidia-ml-py` and prefer `nvitop[cudaXX]` instead by [@XuehaiPan](https://github.com/XuehaiPan) in [#179](https://github.com/XuehaiPan/nvitop/pull/179).

------

## [1.5.2] - 2025-07-25

### Changed

- Minor tweak display for device name in TUI by [@XuehaiPan](https://github.com/XuehaiPan).

### Fixed

- Fix snapshot cache for GPU processes with shared host process by [@XuehaiPan](https://github.com/XuehaiPan) in [#172](https://github.com/XuehaiPan/nvitop/pull/172).

------

## [1.5.1] - 2025-05-26

### Added

- Add `docker-compose` template for `nvitop-exporter` by [@gianfranco-s](https://github.com/gianfranco-s) in [#159](https://github.com/XuehaiPan/nvitop/pull/159).

### Fixed

- Fix one-time output rendering on exit for TUI by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [1.5.0] - 2025-04-25

### Added

- Show `%GMBW` in main screen by [@XuehaiPan](https://github.com/XuehaiPan) in [#156](https://github.com/XuehaiPan/nvitop/pull/156).
- Add doctests and add type annotations in `nvitop.tui` by [@XuehaiPan](https://github.com/XuehaiPan) in [#164](https://github.com/XuehaiPan/nvitop/pull/164).

### Changed

- Deprecate `nvitop.callbacks` as officially unmaintained by [@XuehaiPan](https://github.com/XuehaiPan) in [#157](https://github.com/XuehaiPan/nvitop/pull/157).

### Fixed

- Ignore errors when collecting host metrics for host panel by [@XuehaiPan](https://github.com/XuehaiPan) in [#163](https://github.com/XuehaiPan/nvitop/pull/163).

### Removed

- Drop Python 3.7 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#150](https://github.com/XuehaiPan/nvitop/pull/150).

------

## [1.4.2] - 2025-01-27

### Removed

- Vendor third-party dependency `termcolor` by [@XuehaiPan](https://github.com/XuehaiPan) in [#148](https://github.com/XuehaiPan/nvitop/pull/148).
- Remove third-party dependency `cachetools` by [@XuehaiPan](https://github.com/XuehaiPan) in [#147](https://github.com/XuehaiPan/nvitop/pull/147).

------

## [1.4.1] - 2025-01-13

### Fixed

- Fix passing invalid device handle (e.g., GPU is lost) to NVML functions by [@XuehaiPan](https://github.com/XuehaiPan) in [#146](https://github.com/XuehaiPan/nvitop/pull/146).
- Fix CUDA device selection tool `nvisel` by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [1.4.0] - 2024-12-29

### Added

- Add Grafana dashboard for `nvitop-exporter` by [@XuehaiPan](https://github.com/XuehaiPan) in [#138](https://github.com/XuehaiPan/nvitop/pull/138).
- Handle exceptions for function `getpass.getuser()` by [@XuehaiPan](https://github.com/XuehaiPan) in [#130](https://github.com/XuehaiPan/nvitop/pull/130). Issued by [@landgraf](https://github.com/landgraf).

### Changed

- Refactor setup scripts by [@XuehaiPan](https://github.com/XuehaiPan).

### Fixed

- Fix documentation for the `ResourceMetricCollector.clear()` method by [@MyGodItsFull0fStars](https://github.com/MyGodItsFull0fStars) in [#132](https://github.com/XuehaiPan/nvitop/pull/132).
- Gracefully ignore UTF-8 decoding errors by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [1.3.2] - 2023-10-17

### Added

- Add separate implementation for `GpuStatsLogger` callback for `lightning` by [@XuehaiPan](https://github.com/XuehaiPan) in [#114](https://github.com/XuehaiPan/nvitop/pull/114).
- Remove metrics if process is gone in `nvitop-exporter` by [@XuehaiPan](https://github.com/XuehaiPan) in [#107](https://github.com/XuehaiPan/nvitop/pull/107).

------

## [1.3.1] - 2023-10-05

### Added

- Add Python 3.12 classifiers by [@XuehaiPan](https://github.com/XuehaiPan) in [#101](https://github.com/XuehaiPan/nvitop/pull/101).

### Fixed

- Fix `libcuda.cuDeviceGetUuid()` when the UUID contains `0x00` by [@XuehaiPan](https://github.com/XuehaiPan) in [#100](https://github.com/XuehaiPan/nvitop/pull/100).

------

## [1.3.0] - 2023-08-27

### Added

- Add Prometheus exporter by [@XuehaiPan](https://github.com/XuehaiPan) in [#92](https://github.com/XuehaiPan/nvitop/pull/92).
- Add device APIs to query PCIe and NVLink throughput by [@XuehaiPan](https://github.com/XuehaiPan) in [#87](https://github.com/XuehaiPan/nvitop/pull/87).

### Changed

- Use recent timestamp for GPU process utilization query for more accurate per-process GPU usage by [@XuehaiPan](https://github.com/XuehaiPan) in [#85](https://github.com/XuehaiPan/nvitop/pull/85). We extend our heartfelt gratitude to [@2581543189](https://github.com/2581543189) for their invaluable assistance. Their timely comments and comprehensive feedback have greatly contributed to the improvement of this project.

### Fixed

- Fix upstream changes for process info v3 APIs on 535.104.05 driver by [@XuehaiPan](https://github.com/XuehaiPan) in [#94](https://github.com/XuehaiPan/nvitop/pull/94).
- Fix removal for process info v3 APIs on the upstream 535.98 driver by [@XuehaiPan](https://github.com/XuehaiPan) in [#89](https://github.com/XuehaiPan/nvitop/pull/89).

------

## [1.2.0] - 2023-07-24

### Added

- Include last snapshot metrics in the log results for `ResourceMetricCollector` by [@XuehaiPan](https://github.com/XuehaiPan) in [#80](https://github.com/XuehaiPan/nvitop/pull/80).
- Add `mypy` integration and update type annotations by [@XuehaiPan](https://github.com/XuehaiPan) in [#73](https://github.com/XuehaiPan/nvitop/pull/73).

### Fixed

- Fix process info support for NVIDIA R535 driver (CUDA 12.2+) by [@XuehaiPan](https://github.com/XuehaiPan) in [#79](https://github.com/XuehaiPan/nvitop/pull/79).
- Fix inappropriate exception catching in function `libcuda.cuDeviceGetUuid` by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [1.1.2] - 2023-04-11

### Fixed

- Further isolate the `CUDA_VISIBLE_DEVICES` parser in a subprocess by [@XuehaiPan](https://github.com/XuehaiPan) in [#70](https://github.com/XuehaiPan/nvitop/pull/70).

------

## [1.1.1] - 2023-04-07

### Fixed

- Fix MIG device support by [@XuehaiPan](https://github.com/XuehaiPan).

------

## [1.1.0] - 2023-04-07

### Added

- Support float number as snapshot interval that >= 0.25s by [@XuehaiPan](https://github.com/XuehaiPan) in [#67](https://github.com/XuehaiPan/nvitop/pull/67).
- Show more host metrics (e.g., used virtual memory, uptime) in CLI by [@XuehaiPan](https://github.com/XuehaiPan) in [#59](https://github.com/XuehaiPan/nvitop/pull/59).

### Changed

- Move `TTLCache` usage to CLI-only by [@XuehaiPan](https://github.com/XuehaiPan) in [#66](https://github.com/XuehaiPan/nvitop/pull/66).

### Fixed

- Respect `FORCE_COLOR` and `NO_COLOR` environment variables by [@XuehaiPan](https://github.com/XuehaiPan).

### Removed

- Drop Python 3.6 support by [@XuehaiPan](https://github.com/XuehaiPan) in [#56](https://github.com/XuehaiPan/nvitop/pull/56).

------

## [1.0.0] - 2023-02-01

### Added

- The first stable release of `nvitop` by [@XuehaiPan](https://github.com/XuehaiPan).

------

[Unreleased]: https://github.com/XuehaiPan/nvitop/compare/v1.5.3...HEAD
[1.5.3]: https://github.com/XuehaiPan/nvitop/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/XuehaiPan/nvitop/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/XuehaiPan/nvitop/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/XuehaiPan/nvitop/compare/v1.4.2...v1.5.0
[1.4.2]: https://github.com/XuehaiPan/nvitop/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/XuehaiPan/nvitop/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/XuehaiPan/nvitop/compare/v1.3.2...v1.4.0
[1.3.2]: https://github.com/XuehaiPan/nvitop/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/XuehaiPan/nvitop/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/XuehaiPan/nvitop/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/XuehaiPan/nvitop/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/XuehaiPan/nvitop/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/XuehaiPan/nvitop/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/XuehaiPan/nvitop/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/XuehaiPan/nvitop/releases/tag/v1.0.0
