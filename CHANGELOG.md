# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------

## [Unreleased]

### Added

- Add device APIs to query PCIe and NVLink throughput by [@XuehaiPan](https://github.com/XuehaiPan) in [#87](https://github.com/XuehaiPan/nvitop/pull/87).

### Changed

- Use recent timestamp for GPU process utilization query for more accurate per-process GPU usage by [@XuehaiPan](https://github.com/XuehaiPan) in [#85](https://github.com/XuehaiPan/nvitop/pull/85). We extend our heartfelt gratitude to [@2581543189](https://github.com/2581543189) for their invaluable assistance. Their timely comments and comprehensive feedback have greatly contributed to the improvement of this project.

### Fixed

- Fix upstream changes for process info v3 APIs on 535.104.05 driver by [@XuehaiPan](https://github.com/XuehaiPan) in [#94](https://github.com/XuehaiPan/nvitop/pull/94).
- Fix removal for process info v3 APIs on the upstream 535.98 driver by [@XuehaiPan](https://github.com/XuehaiPan) in [#89](https://github.com/XuehaiPan/nvitop/pull/89).

### Removed

-

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

[Unreleased]: https://github.com/XuehaiPan/nvitop/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/XuehaiPan/nvitop/releases/tag/v1.2.0
[1.1.2]: https://github.com/XuehaiPan/nvitop/releases/tag/v1.1.2
[1.1.1]: https://github.com/XuehaiPan/nvitop/releases/tag/v1.1.1
[1.1.0]: https://github.com/XuehaiPan/nvitop/releases/tag/v1.1.0
[1.0.0]: https://github.com/XuehaiPan/nvitop/releases/tag/v1.0.0
