# nvitop

<!-- markdownlint-disable html -->

![Python 3.5+](https://img.shields.io/badge/Python-3.5%2B-brightgreen)
[![PyPI](https://img.shields.io/pypi/v/nvitop?label=pypi&logo=pypi)](https://pypi.org/project/nvitop)
![Status](https://img.shields.io/pypi/status/nvitop?label=status)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/nvitop?label=conda&logo=condaforge)](https://anaconda.org/conda-forge/nvitop)
[![Documentation Status](https://img.shields.io/readthedocs/nvitop?label=docs&logo=readthedocs)](https://nvitop.readthedocs.io)
[![Downloads](https://static.pepy.tech/personalized-badge/nvitop?period=total&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/nvitop)
[![GitHub Repo Stars](https://img.shields.io/github/stars/XuehaiPan/nvitop?label=stars&logo=github&color=brightgreen)](https://github.com/XuehaiPan/nvitop/stargazers)
[![License](https://img.shields.io/github/license/XuehaiPan/nvitop?label=license)](#license)

An interactive NVIDIA-GPU process viewer, the one-stop solution for GPU process management. The full API references host at <https://nvitop.readthedocs.io>.

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/171005261-1aad126e-dc27-4ed3-a89b-7f9c1c998bf7.png" alt="Monitor">
  <br/>
  Monitor mode of <code>nvitop</code>.
  <br/>
  (TERM: GNOME Terminal / OS: Ubuntu 16.04 LTS (over SSH) / Locale: <code>en_US.UTF-8</code>)
</p>

### Table of Contents  <!-- omit in toc --> <!-- markdownlint-disable heading-increment -->

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Device and Process Status](#device-and-process-status)
  - [Resource Monitor](#resource-monitor)
    - [For Docker Users](#for-docker-users)
    - [For SSH Users](#for-ssh-users)
    - [Command Line Options and Environment Variables](#command-line-options-and-environment-variables)
    - [Keybindings for Monitor Mode](#keybindings-for-monitor-mode)
  - [CUDA Visible Devices Selection Tool](#cuda-visible-devices-selection-tool)
  - [Callback Functions for Machine Learning Frameworks](#callback-functions-for-machine-learning-frameworks)
    - [Callback for TensorFlow (Keras)](#callback-for-tensorflow-keras)
    - [Callback for PyTorch Lightning](#callback-for-pytorch-lightning)
    - [TensorBoard Integration](#tensorboard-integration)
  - [More than a Monitor](#more-than-a-monitor)
    - [Quick Start](#quick-start)
    - [Status Snapshot](#status-snapshot)
    - [Resource Metric Collector](#resource-metric-collector)
    - [Low-level APIs](#low-level-apis)
      - [Device](#device)
      - [Process](#process)
      - [Host (inherited from psutil)](#host-inherited-from-psutil)
- [Screenshots](#screenshots)
- [License](#license)

------

This project is inspired by [nvidia-htop](https://github.com/peci1/nvidia-htop) and [nvtop](https://github.com/Syllo/nvtop) for monitoring, and [gpustat](https://github.com/wookayin/gpustat) for application integration.

[nvidia-htop](https://github.com/peci1/nvidia-htop) is a tool for enriching the output of `nvidia-smi`. It uses regular expressions to read the output of `nvidia-smi` from a subprocess, which is inefficient. In the meanwhile, there is a powerful interactive GPU monitoring tool called [nvtop](https://github.com/Syllo/nvtop). But [nvtop](https://github.com/Syllo/nvtop) is written in *C*, which makes it lack of portability. And what is really inconvenient is that you should compile it yourself during the installation. Therefore, I made this repo. I got a lot help when reading the source code of [ranger](https://github.com/ranger/ranger), the console file manager. Some files in this repo are modified from [ranger](https://github.com/ranger/ranger) under the **GPLv3 License**.

If this repo is useful to you, please star ⭐️ it to let more people know 🤗. [![GitHub Repo Stars](https://img.shields.io/github/stars/XuehaiPan/nvitop?label=stars&logo=github&color=brightgreen)](https://github.com/XuehaiPan/nvitop)

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/202362686-859bf4ad-6237-46ca-b2f7-f547d2f63213.png" alt="Comparison">
  <br/>
  Compare to <code>nvidia-smi</code>.
</p>

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/202362811-34f2c01d-97c8-49d2-b19b-0d7da648f2d5.png" alt="Filter">
  <br/>
  Process filtering and more colorful interface.
</p>

------

## Features

- **Informative and fancy output**: show more information than `nvidia-smi` with colorized fancy box drawing.
- **Monitor mode**: can run as a resource monitor, rather than print the results only once. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop), limited support with command `watch -c`)
  - bar charts and history graphs
  - process sorting
  - process filtering
  - send signals to processes with a keystroke
  - tree-view screen for GPU processes and their parent processes
  - environment variable screen
  - help screen
  - mouse support
- **Interactive**: responsive for user input (from keyboard and/or mouse) in monitor mode. (vs. [gpustat](https://github.com/wookayin/gpustat) & [py3nvml](https://github.com/fbcotter/py3nvml))
- **Efficient**:
  - query device status using [*NVML Python bindings*](https://pypi.org/project/nvidia-ml-py) directly, instead of parsing the output of `nvidia-smi`. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop))
  - cache results with `TTLCache` from [cachetools](https://github.com/tkem/cachetools). (vs. [gpustat](https://github.com/wookayin/gpustat))
  - display information using the `curses` library rather than `print` with ANSI escape codes. (vs. [py3nvml](https://github.com/fbcotter/py3nvml))
  - asynchronously gather information using multi-threading and correspond to user input much faster. (vs. [nvtop](https://github.com/Syllo/nvtop))
- **Portable**: work on both Linux and Windows.
  - get host process information using the cross-platform library [psutil](https://github.com/giampaolo/psutil) instead of calling `ps -p <pid>` in a subprocess. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop) & [py3nvml](https://github.com/fbcotter/py3nvml))
  - written in pure Python, easy to install with `pip`. (vs. [nvtop](https://github.com/Syllo/nvtop))
- **Integrable**: easy to integrate into other applications, more than monitoring. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop) & [nvtop](https://github.com/Syllo/nvtop))

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/129374533-fe06c01a-630d-4994-b54b-821cccd0d33c.png" alt="Windows">
  <br/>
  <code>nvitop</code> supports Windows!
  <br/>
  (SHELL: PowerShell / TERM: Windows Terminal / OS: Windows 10 / Locale: <code>en-US</code>)
</p>

------

## Requirements

- Python 3.5+ (with `pip>=10.0`)
- NVIDIA Management Library (NVML)
- nvidia-ml-py
- psutil
- cachetools
- termcolor
- curses<sup>[*](#curses)</sup> (with `libncursesw`)

**NOTE:** The [NVIDIA Management Library (*NVML*)](https://developer.nvidia.com/nvidia-management-library-nvml) is a C-based programmatic interface for monitoring and managing various states. The runtime version of NVML library ships with the NVIDIA display driver (available at [Download Drivers | NVIDIA](https://www.nvidia.com/Download/index.aspx)), or can be downloaded as part of the NVIDIA CUDA Toolkit (available at [CUDA Toolkit | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)). The lists of OS platforms and NVIDIA-GPUs supported by the NVML library can be found in the [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html).

This repository contains a Bash script to install/upgrade the NVIDIA drivers for Ubuntu Linux. For example:

```bash
git clone --depth=1 https://github.com/XuehaiPan/nvitop.git && cd nvitop

# Change to tty3 console (required for desktop users with GUI (tty2))
# Optional for SSH users
sudo chvt 3  # or use keyboard shortcut: Ctrl-LeftAlt-F3

bash install-nvidia-driver.sh --package=nvidia-driver-470  # install the R470 driver from ppa:graphics-drivers
bash install-nvidia-driver.sh --latest                     # install the latest driver from ppa:graphics-drivers
```

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/174480112-e9a35edc-8f42-438e-a103-1d0ce998b381.png" alt="install-nvidia-driver">
  <br/>
  NVIDIA driver installer for Ubuntu Linux.
</p>

Run `bash install-nvidia-driver.sh --help` for more information.

<a name="curses">*</a> The `curses` library is a built-in module of Python on Unix-like systems, and it is supported by a third-party package called `windows-curses` on Windows using PDCurses. Inconsistent behavior of `nvitop` may occur on different terminal emulators on Windows, such as missing mouse support.

------

## Installation

**It is highly recommended to install `nvitop` in an isolated virtual environment.** Simple installation and run via [`pipx`](https://pypa.github.io/pipx):

```bash
pipx run nvitop
```

Install from PyPI ([![PyPI](https://img.shields.io/pypi/v/nvitop?label=pypi&logo=pypi)](https://pypi.org/project/nvitop) / ![Status](https://img.shields.io/pypi/status/nvitop?label=status)):

```bash
pip3 install --upgrade nvitop
```

Install from conda-forge ([![conda-forge](https://img.shields.io/conda/v/conda-forge/nvitop?logo=condaforge)](https://anaconda.org/conda-forge/nvitop)):

```bash
conda install -c conda-forge nvitop
```

Install the latest version from GitHub (![Commit Count](https://img.shields.io/github/commits-since/XuehaiPan/nvitop/v0.10.2)):

```bash
pip3 install git+https://github.com/XuehaiPan/nvitop.git#egg=nvitop
```

Or, clone this repo and install manually:

```bash
git clone --depth=1 https://github.com/XuehaiPan/nvitop.git
cd nvitop
pip3 install .
```

**NOTE:** If you encounter the *"nvitop: command not found"* error after installation, please check whether you have added the Python console script path (e.g., `"${HOME}/.local/bin"`) to your `PATH` environment variable. Alternatively, you can use `python3 -m nvitop`.

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/178963038-a5cd4eb5-02a8-4456-966f-d5ff04eb44d8.png" alt="MIG Device Support">
  <br/>
  MIG Device Support.
  <br/>
</p>

------

## Usage

### Device and Process Status

Query the device and process status. The output is similar to `nvidia-smi`, but has been enriched and colorized.

```bash
# Query status of all devices
$ nvitop -1  # or use `python3 -m nvitop -1`

# Specify query devices (by integer indices)
$ nvitop -1 -o 0 1  # only show <GPU 0> and <GPU 1>

# Only show devices in `CUDA_VISIBLE_DEVICES` (by integer indices or UUID strings)
$ nvitop -1 -ov

# Only show GPU processes with the compute context (type: 'C' or 'C+G')
$ nvitop -1 -c
```

When the `-1` switch is on, the result will be displayed **ONLY ONCE** (same as the default behavior of `nvidia-smi`). This is much faster and has lower resource usage. See [Command Line Options](#command-line-options-and-environment-variables) for more command options.

There is also a CLI tool called `nvisel` that ships with the `nvitop` PyPI package. See [CUDA Visible Devices Selection Tool](#cuda-visible-devices-selection-tool) for more information.

### Resource Monitor

Run as a resource monitor:

```bash
# Monitor mode (when the display mode is omitted, `NVITOP_MONITOR_MODE` will be used)
$ nvitop  # or use `python3 -m nvitop`

# Automatically configure the display mode according to the terminal size
$ nvitop -m auto     # shortcut: `a` key

# Arbitrarily display as `full` mode
$ nvitop -m full     # shortcut: `f` key

# Arbitrarily display as `compact` mode
$ nvitop -m compact  # shortcut: `c` key

# Specify query devices (by integer indices)
$ nvitop -o 0 1  # only show <GPU 0> and <GPU 1>

# Only show devices in `CUDA_VISIBLE_DEVICES` (by integer indices or UUID strings)
$ nvitop -ov

# Only show GPU processes with the compute context (type: 'C' or 'C+G')
$ nvitop -c

# Use ASCII characters only
$ nvitop -U  # useful for terminals without Unicode support

# For light terminals
$ nvitop --light

# For spectrum-like bar charts (requires the terminal supports 256-color)
$ nvitop --colorful
```

You can configure the default monitor mode with the `NVITOP_MONITOR_MODE` environment variable (default `auto` if not set). See [Command Line Options and Environment Variables](#command-line-options-and-environment-variables) for more command options.

In monitor mode, you can use <kbd>Ctrl-c</kbd> / <kbd>T</kbd> / <kbd>K</kbd> keys to interrupt / terminate / kill a process. And it's recommended to *terminate* or *kill* a process in the **tree-view screen** (shortcut: <kbd>t</kbd>). For normal users, `nvitop` will shallow other users' processes (in low-intensity colors). For **system administrators**, you can use `sudo nvitop` to terminate other users' processes.

Also, to enter the process metrics screen, select a process and then press the <kbd>Enter</kbd> / <kbd>Return</kbd> key . `nvitop` dynamically displays the process metrics with live graphs.

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/192108815-37c03705-be44-47d4-9908-6d05175db230.png" alt="Process Metrics Screen">
  <br/>
  Watch metrics for a specific process (shortcut: <kbd>Enter</kbd> / <kbd>Return</kbd>).
</p>

Press <kbd>h</kbd> for help or <kbd>q</kbd> to return to the terminal. See [Keybindings for Monitor Mode](#keybindings-for-monitor-mode) for more shortcuts.

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/192108664-61f1983c-6f62-48e6-87c5-29633d9c409e.png" alt="Help Screen">
  <br/>
  <code>nvitop</code> comes with a help screen (shortcut: <kbd>h</kbd>).
</p>

#### For Docker Users

Build and run the Docker image using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker):

```bash
git clone --depth=1 https://github.com/XuehaiPan/nvitop.git && cd nvitop  # clone this repo first
docker build --tag nvitop:latest .  # build the Docker image
docker run -it --rm --runtime=nvidia --gpus=all --pid=host nvitop:latest  # run the Docker container
```

The [`Dockerfile`](Dockerfile) has a optional build argument `basetag` (default: `418.87.01-ubuntu18.04`) for the tag of image [`nvidia/driver`](https://hub.docker.com/r/nvidia/driver/tags).

**NOTE:** Don't forget to add the `--pid=host` option when running the container.

#### For SSH Users

Run `nvitop` directly on the SSH session instead of a login shell:

```bash
ssh user@host -t nvitop                 # installed by `sudo pip3 install ...`
ssh user@host -t '~/.local/bin/nvitop'  # installed by `pip3 install --user ...`
```

**NOTE:** Users need to add the `-t` option to allocate a pseudo-terminal over the SSH session for monitor mode.

#### Command Line Options and Environment Variables

Type `nvitop --help` for more command options:

```text
usage: nvitop [--help] [--version] [--once] [--monitor [{auto,full,compact}]]
              [--interval SEC] [--ascii] [--colorful] [--force-color] [--light]
              [--gpu-util-thresh th1 th2] [--mem-util-thresh th1 th2]
              [--only idx [idx ...]] [--only-visible]
              [--compute] [--only-compute] [--graphics] [--only-graphics]
              [--user [USERNAME ...]] [--pid PID [PID ...]]

An interactive NVIDIA-GPU process viewer.

optional arguments:
  --help, -h            Show this help message and exit.
  --version, -V         Show nvitop's version number and exit.
  --once, -1            Report query data only once.
  --monitor [{auto,full,compact}], -m [{auto,full,compact}]
                        Run as a resource monitor. Continuously report query data and handle user inputs.
                        If the argument is omitted, the value from `NVITOP_MONITOR_MODE` will be used.
                        (default fallback mode: auto)
  --interval SEC        Process status update interval in seconds. (default: 2)
  --ascii, --no-unicode, -U
                        Use ASCII characters only, which is useful for terminals without Unicode support.

coloring:
  --colorful            Use gradient colors to get spectrum-like bar charts. This option is only available
                        when the terminal supports 256 colors. You may need to set environment variable
                        `TERM="xterm-256color"`. Note that the terminal multiplexer, such as `tmux`, may
                        override the `TREM` variable.
  --force-color         Force colorize even when `stdout` is not a TTY terminal.
  --light               Tweak visual results for light theme terminals in monitor mode.
                        Set variable `NVITOP_MONITOR_MODE="light"` on light terminals for convenience.
  --gpu-util-thresh th1 th2
                        Thresholds of GPU utilization to determine the load intensity.
                        Coloring rules: light < th1 % <= moderate < th2 % <= heavy.
                        ( 1 <= th1 < th2 <= 99, defaults: 10 75 )
  --mem-util-thresh th1 th2
                        Thresholds of GPU memory percent to determine the load intensity.
                        Coloring rules: light < th1 % <= moderate < th2 % <= heavy.
                        ( 1 <= th1 < th2 <= 99, defaults: 10 80 )

device filtering:
  --only idx [idx ...], -o idx [idx ...]
                        Only show the specified devices, suppress option `--only-visible`.
  --only-visible, -ov   Only show devices in the `CUDA_VISIBLE_DEVICES` environment variable.

process filtering:
  --compute, -c         Only show GPU processes with the compute context. (type: 'C' or 'C+G')
  --only-compute, -C    Only show GPU processes exactly with the compute context. (type: 'C' only)
  --graphics, -g        Only show GPU processes with the graphics context. (type: 'G' or 'C+G')
  --only-graphics, -G   Only show GPU processes exactly with the graphics context. (type: 'G' only)
  --user [USERNAME ...], -u [USERNAME ...]
                        Only show processes of the given users (or `$USER` for no argument).
  --pid PID [PID ...], -p PID [PID ...]
                        Only show processes of the given PIDs.
```

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/182555606-8388e5a5-43a9-4990-90d4-46e45ac448a0.png" alt="Spectrum-like Bar Charts">
  <br/>
  Spectrum-like bar charts (with option <code>--colorful</code>).
</p>

`nvitop` can accept the following environment variables for monitor mode:

| Name                                   | Description                                         | Valid Values                                                            | Default Value     |
| -------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------- | ----------------- |
| `NVITOP_MONITOR_MODE`                  | The default display mode (a comma-separated string) | `auto` / `full` / `compact`<br>`plain` / `colorful`<br>`dark` / `light` | `auto,plain,dark` |
| `NVITOP_GPU_UTILIZATION_THRESHOLDS`    | Thresholds of GPU utilization                       | `10,75` , `1,99`, ...                                                   | `10,75`           |
| `NVITOP_MEMORY_UTILIZATION_THRESHOLDS` | Thresholds of GPU memory percent                    | `10,80` , `1,99`, ...                                                   | `10,80`           |
| `LOGLEVEL`                             | Log level for log messages                          | `DEBUG` , `INFO`, `WARNING`, ...                                        | `WARNING`         |

For example:

```bash
# Replace the following export statements if you are not using Bash / Zsh
export NVITOP_MONITOR_MODE="full,light"

# Full monitor mode with light terminal tweaks
nvitop
```

For convenience, you can add these environment variables to your shell startup file, e.g.:

```bash
# For Bash
echo 'export NVITOP_MONITOR_MODE="full"' >> ~/.bashrc

# For Zsh
echo 'export NVITOP_MONITOR_MODE="full"' >> ~/.zshrc

# For Fish
echo 'set -gx NVITOP_MONITOR_MODE "full"' >> ~/.config/fish/config.fish

# For PowerShell
'$Env:NVITOP_MONITOR_MODE = "full"' >> $PROFILE.CurrentUserAllHosts
```

#### Keybindings for Monitor Mode

|                                                                        Key | Binding                                                                              |
| -------------------------------------------------------------------------: | :----------------------------------------------------------------------------------- |
|                                                                        `q` | Quit and return to the terminal.                                                     |
|                                                                  `h` / `?` | Go to the help screen.                                                               |
|                                                            `a` / `f` / `c` | Change the display mode to *auto* / *full* / *compact*.                              |
|                                                     `r` / `<C-r>` / `<F5>` | Force refresh the window.                                                            |
|                                                                            |                                                                                      |
| `<Up>` / `<Down>`<br>`<A-k>` / `<A-j>`<br>`<Tab>` / `<S-Tab>`<br>`<Wheel>` | Select and highlight a process.                                                      |
|                   `<Left>` / `<Right>`<br>`<A-h>` / `<A-l>`<br>`<S-Wheel>` | Scroll the host information of processes.                                            |
|                                                                   `<Home>` | Select the first process.                                                            |
|                                                                    `<End>` | Select the last process.                                                             |
|                                                             `<C-a>`<br>`^` | Scroll left to the beginning of the process entry (i.e. beginning of line).          |
|                                                             `<C-e>`<br>`$` | Scroll right to the end of the process entry (i.e. end of line).                     |
|              `<PageUp>` / `<PageDown>`<br/> `<A-K>` / `<A-J>`<br>`[` / `]` | scroll entire screen (for large amounts of processes).                               |
|                                                                            |                                                                                      |
|                                                                  `<Space>` | Tag/untag current process.                                                           |
|                                                                    `<Esc>` | Clear process selection.                                                             |
|                                                             `<C-c>`<br>`I` | Send `signal.SIGINT` to the selected process (interrupt).                            |
|                                                                        `T` | Send `signal.SIGTERM` to the selected process (terminate).                           |
|                                                                        `K` | Send `signal.SIGKILL` to the selected process (kill).                                |
|                                                                            |                                                                                      |
|                                                                        `e` | Show process environment.                                                            |
|                                                                        `t` | Toggle tree-view screen.                                                             |
|                                                                  `<Enter>` | Toggle tree-view screen.                                                             |
|                                                                            |                                                                                      |
|                                                                  `,` / `.` | Select the sort column.                                                              |
|                                                                        `/` | Reverse the sort order.                                                              |
|                                                                `on` (`oN`) | Sort processes in the natural order, i.e., in ascending (descending) order of `GPU`. |
|                                                                `ou` (`oU`) | Sort processes by `USER` in ascending (descending) order.                            |
|                                                                `op` (`oP`) | Sort processes by `PID` in descending (ascending) order.                             |
|                                                                `og` (`oG`) | Sort processes by `GPU-MEM` in descending (ascending) order.                         |
|                                                                `os` (`oS`) | Sort processes by `%SM` in descending (ascending) order.                             |
|                                                                `oc` (`oC`) | Sort processes by `%CPU` in descending (ascending) order.                            |
|                                                                `om` (`oM`) | Sort processes by `%MEM` in descending (ascending) order.                            |
|                                                                `ot` (`oT`) | Sort processes by `TIME` in descending (ascending) order.                            |

**HINT:** It's recommended to terminate or kill a process in the tree-view screen (shortcut: <kbd>t</kbd>).

------

### CUDA Visible Devices Selection Tool

Automatically select `CUDA_VISIBLE_DEVICES` from the given criteria. Example usage of the CLI tool:

```console
# All devices but sorted
$ nvisel       # or use `python3 -m nvitop.select`
6,5,4,3,2,1,0,7,8

# A simple example to select 4 devices
$ nvisel -n 4  # or use `python3 -m nvitop.select -n 4`
6,5,4,3

# Select available devices that satisfy the given constraints
$ nvisel --min-count 2 --max-count 3 --min-free-memory 5GiB --max-gpu-utilization 60
6,5,4

# Set `CUDA_VISIBLE_DEVICES` environment variable using `nvisel`
$ export CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="$(nvisel -c 1 -f 10GiB)"
CUDA_VISIBLE_DEVICES="6,5,4,3,2,1,0"

# Use UUID strings in `CUDA_VISIBLE_DEVICES` environment variable
$ export CUDA_VISIBLE_DEVICES="$(nvisel -O uuid -c 2 -f 5000M)"
CUDA_VISIBLE_DEVICES="GPU-849d5a8d-610e-eeea-1fd4-81ff44a23794,GPU-18ef14e9-dec6-1d7e-1284-3010c6ce98b1,GPU-96de99c9-d68f-84c8-424c-7c75e59cc0a0,GPU-2428d171-8684-5b64-830c-435cd972ec4a,GPU-6d2a57c9-7783-44bb-9f53-13f36282830a,GPU-f8e5a624-2c7e-417c-e647-b764d26d4733,GPU-f9ca790e-683e-3d56-00ba-8f654e977e02"

# Pipe output to other shell utilities
$ nvisel --newline -O uuid -C 6 -f 8GiB
GPU-849d5a8d-610e-eeea-1fd4-81ff44a23794
GPU-18ef14e9-dec6-1d7e-1284-3010c6ce98b1
GPU-96de99c9-d68f-84c8-424c-7c75e59cc0a0
GPU-2428d171-8684-5b64-830c-435cd972ec4a
GPU-6d2a57c9-7783-44bb-9f53-13f36282830a
GPU-f8e5a624-2c7e-417c-e647-b764d26d4733
$ nvisel -0 -O uuid -c 2 -f 4GiB | xargs -0 -I {} nvidia-smi --id={} --query-gpu=index,memory.free --format=csv
CUDA_VISIBLE_DEVICES="GPU-849d5a8d-610e-eeea-1fd4-81ff44a23794,GPU-18ef14e9-dec6-1d7e-1284-3010c6ce98b1,GPU-96de99c9-d68f-84c8-424c-7c75e59cc0a0,GPU-2428d171-8684-5b64-830c-435cd972ec4a,GPU-6d2a57c9-7783-44bb-9f53-13f36282830a,GPU-f8e5a624-2c7e-417c-e647-b764d26d4733,GPU-f9ca790e-683e-3d56-00ba-8f654e977e02"
index, memory.free [MiB]
6, 11018 MiB
index, memory.free [MiB]
5, 11018 MiB
index, memory.free [MiB]
4, 11018 MiB
index, memory.free [MiB]
3, 11018 MiB
index, memory.free [MiB]
2, 11018 MiB
index, memory.free [MiB]
1, 11018 MiB
index, memory.free [MiB]
0, 11018 MiB

# Normalize the `CUDA_VISIBLE_DEVICES` environment variable (e.g. convert UUIDs to indices or get full UUIDs for an abbreviated form)
$ nvisel -i "GPU-18ef14e9,GPU-849d5a8d" -S
5,6
$ nvisel -i "GPU-18ef14e9,GPU-849d5a8d" -S -O uuid --newline
GPU-18ef14e9-dec6-1d7e-1284-3010c6ce98b1
GPU-849d5a8d-610e-eeea-1fd4-81ff44a23794
```

You can also integrate `nvisel` into your training script like this:

```python
# Put this at the top of the Python script
import os
from nvitop import select_devices

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    select_devices(format='uuid', min_count=4, min_free_memory='8GiB')
)
```

Type `nvisel --help` for more command options:

```text
usage: nvisel [--help] [--version]
              [--inherit [CUDA_VISIBLE_DEVICES]] [--account-as-free [USERNAME ...]]
              [--min-count N] [--max-count N] [--count N]
              [--min-free-memory SIZE] [--min-total-memory SIZE]
              [--max-gpu-utilization RATE] [--max-memory-utilization RATE]
              [--tolerance TOL]
              [--format FORMAT] [--sep SEP | --newline | --null] [--no-sort]

CUDA visible devices selection tool.

optional arguments:
  --help, -h            Show this help message and exit.
  --version, -V         Show nvisel's version number and exit.

constraints:
  --inherit [CUDA_VISIBLE_DEVICES], -i [CUDA_VISIBLE_DEVICES]
                        Inherit the given `CUDA_VISIBLE_DEVICES`. If the argument is omitted, use the
                        value from the environment. This means selecting a subset of the currently
                        CUDA-visible devices.
  --account-as-free [USERNAME ...]
                        Account the used GPU memory of the given users as free memory.
                        If this option is specified but without argument, `$USER` will be used.
  --min-count N, -c N   Minimum number of devices to select. (default: 0)
                        The tool will fail (exit non-zero) if the requested resource is not available.
  --max-count N, -C N   Maximum number of devices to select. (default: all devices)
  --count N, -n N       Overriding both `--min-count N` and `--max-count N`.
  --min-free-memory SIZE, -f SIZE
                        Minimum free memory of devices to select. (example value: 4GiB)
                        If this constraint is given, check against all devices.
  --min-total-memory SIZE, -t SIZE
                        Minimum total memory of devices to select. (example value: 10GiB)
                        If this constraint is given, check against all devices.
  --max-gpu-utilization RATE, -G RATE
                        Maximum GPU utilization rate of devices to select. (example value: 30)
                        If this constraint is given, check against all devices.
  --max-memory-utilization RATE, -M RATE
                        Maximum memory bandwidth utilization rate of devices to select. (example value: 50)
                        If this constraint is given, check against all devices.
  --tolerance TOL, --tol TOL
                        The constraints tolerance (in percentage). (default: 0, i.e., strict)
                        This option can loose the constraints if the requested resource is not available.
                        For example, set `--tolerance=20` will accept a device with only 4GiB of free
                        memory when set `--min-free-memory=5GiB`.

formatting:
  --format FORMAT, -O FORMAT
                        The output format of the selected device identifiers. (default: index)
                        If any MIG device found, the output format will be fallback to `uuid`.
  --sep SEP, --separator SEP, -s SEP
                        Separator for the output. (default: ',')
  --newline             Use newline character as separator for the output, equivalent to `--sep=$'\n'`.
  --null, -0            Use null character ('\x00') as separator for the output. This option corresponds
                        to the `-0` option of `xargs`.
  --no-sort, -S         Do not sort the device by memory usage and GPU utilization.
```

------

### Callback Functions for Machine Learning Frameworks

`nvitop` provides two builtin callbacks for [TensorFlow (Keras)](https://www.tensorflow.org) and [PyTorch Lightning](https://pytorchlightning.ai).

#### Callback for [TensorFlow (Keras)](https://www.tensorflow.org)

```python
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.keras.callbacks import TensorBoard
from nvitop.callbacks.keras import GpuStatsLogger
gpus = ['/gpu:0', '/gpu:1']  # or `gpus = [0, 1]` or `gpus = 2`
model = Xception(weights=None, ..)
model = multi_gpu_model(model, gpus)  # optional
model.compile(..)
tb_callback = TensorBoard(log_dir='./logs')  # or `keras.callbacks.CSVLogger`
gpu_stats = GpuStatsLogger(gpus)
model.fit(.., callbacks=[gpu_stats, tb_callback])
```

**NOTE:** Users should assign a `keras.callbacks.TensorBoard` callback or a `keras.callbacks.CSVLogger` callback to the model. And the `GpuStatsLogger` callback should be placed before the `keras.callbacks.TensorBoard` / `keras.callbacks.CSVLogger` callback.

#### Callback for [PyTorch Lightning](https://pytorchlightning.ai)

```python
from pytorch_lightning import Trainer
from nvitop.callbacks.pytorch_lightning import GpuStatsLogger
gpu_stats = GpuStatsLogger()
trainer = Trainer(gpus=[..], logger=True, callbacks=[gpu_stats])
```

**NOTE:** Users should assign a logger to the trainer.

#### [TensorBoard](https://github.com/tensorflow/tensorboard) Integration

Please refer to [Resource Metric Collector](#resource-metric-collector) for an example.

------

### More than a Monitor

`nvitop` can be easily integrated into other applications. You can use `nvitop` to make your own monitoring tools. The full API references host at <https://nvitop.readthedocs.io>.

#### Quick Start

A minimal script to monitor the GPU devices based on APIs from `nvitop`:

```python
from nvitop import Device

devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
for device in devices:
    processes = device.processes()  # type: Dict[int, GpuProcess]
    sorted_pids = sorted(processes.keys())

    print(device)
    print(f'  - Fan speed:       {device.fan_speed()}%')
    print(f'  - Temperature:     {device.temperature()}C')
    print(f'  - GPU utilization: {device.gpu_utilization()}%')
    print(f'  - Total memory:    {device.memory_total_human()}')
    print(f'  - Used memory:     {device.memory_used_human()}')
    print(f'  - Free memory:     {device.memory_free_human()}')
    print(f'  - Processes ({len(processes)}): {sorted_pids}')
    for pid in sorted_pids:
        print(f'    - {processes[pid]}')
    print('-' * 120)
```

Another more advanced approach with coloring:

```python
import time

from nvitop import Device, GpuProcess, NA, colored

print(colored(time.strftime('%a %b %d %H:%M:%S %Y'), color='red', attrs=('bold',)))

devices = Device.cuda.all()  # or `Device.all()` to use NVML ordinal instead
separator = False
for device in devices:
    processes = device.processes()  # type: Dict[int, GpuProcess]

    print(colored(str(device), color='green', attrs=('bold',)))
    print(colored('  - Fan speed:       ', color='blue', attrs=('bold',)) + f'{device.fan_speed()}%')
    print(colored('  - Temperature:     ', color='blue', attrs=('bold',)) + f'{device.temperature()}C')
    print(colored('  - GPU utilization: ', color='blue', attrs=('bold',)) + f'{device.gpu_utilization()}%')
    print(colored('  - Total memory:    ', color='blue', attrs=('bold',)) + f'{device.memory_total_human()}')
    print(colored('  - Used memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_used_human()}')
    print(colored('  - Free memory:     ', color='blue', attrs=('bold',)) + f'{device.memory_free_human()}')
    if len(processes) > 0:
        processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
        processes.sort(key=lambda process: (process.username, process.pid))

        print(colored(f'  - Processes ({len(processes)}):', color='blue', attrs=('bold',)))
        fmt = '    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}  {gpu_memory:>8}  {sm:>3}  {command:<}'.format
        print(colored(fmt(pid='PID', username='USERNAME',
                          cpu='CPU%', host_memory='HOST-MEM', time='TIME',
                          gpu_memory='GPU-MEM', sm='SM%',
                          command='COMMAND'),
                      attrs=('bold',)))
        for snapshot in processes:
            print(fmt(pid=snapshot.pid,
                      username=snapshot.username[:7] + ('+' if len(snapshot.username) > 8 else snapshot.username[7:8]),
                      cpu=snapshot.cpu_percent, host_memory=snapshot.host_memory_human,
                      time=snapshot.running_time_human,
                      gpu_memory=(snapshot.gpu_memory_human if snapshot.gpu_memory_human is not NA else 'WDDM:N/A'),
                      sm=snapshot.gpu_sm_utilization,
                      command=snapshot.command))
    else:
        print(colored('  - No Running Processes', attrs=('bold',)))

    if separator:
        print('-' * 120)
    separator = True
```

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/177041142-fe988d58-6a97-4559-84fd-b51204cf9231.png" alt="Demo">
  <br/>
  An example monitoring script built with APIs from <code>nvitop</code>.
</p>

------

#### Status Snapshot

`nvitop` provides a helper function [`take_snapshots`](https://nvitop.readthedocs.io/en/latest/core/collector.html#nvitop.take_snapshots) to retrieve the status of both GPU devices and GPU processes at once. You can type `help(nvitop.take_snapshots)` in Python REPL for detailed documentation.

```python
In [1]: from nvitop import take_snapshots, Device
   ...: import os
   ...: os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'  # comma-separated integers or UUID strings

In [2]: take_snapshots()  # equivalent to `take_snapshots(Device.all())`
Out[2]:
SnapshotResult(
    devices=[
        DeviceSnapshot(
            real=Device(index=0, ...),
            ...
        ),
        ...
    ],
    gpu_processes=[
        GpuProcessSnapshot(
            real=GpuProcess(pid=xxxxxx, device=Device(index=0, ...), ...),
            ...
        ),
        ...
    ]
)

In [3]: device_snapshots, gpu_process_snapshots = take_snapshots(Device.all())  # type: Tuple[List[DeviceSnapshot], List[GpuProcessSnapshot]]

In [4]: device_snapshots, _ = take_snapshots(gpu_processes=False)  # ignore process snapshots

In [5]: take_snapshots(Device.cuda.all())  # use CUDA device enumeration
Out[5]:
SnapshotResult(
    devices=[
        CudaDeviceSnapshot(
            real=CudaDevice(cuda_index=0, nvml_index=1, ...),
            ...
        ),
        CudaDeviceSnapshot(
            real=CudaDevice(cuda_index=1, nvml_index=0, ...),
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

In [6]: take_snapshots(Device.cuda(1))  # <CUDA 1> only
Out[6]:
SnapshotResult(
    devices=[
        CudaDeviceSnapshot(
            real=CudaDevice(cuda_index=1, nvml_index=0, ...),
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
```

Please refer to section [Low-level APIs](#low-level-apis) for more information.

------

#### Resource Metric Collector

[`ResourceMetricCollector`](https://nvitop.readthedocs.io/en/latest/core/collector.html#nvitop.ResourceMetricCollector) is a class that collects resource metrics for host, GPUs and processes running on the GPUs. All metrics will be collected in an asynchronous manner. You can type `help(nvitop.ResourceMetricCollector)` in Python REPL for detailed documentation.

```python
In [1]: from nvitop import ResourceMetricCollector, Device
   ...: import os
   ...: os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'  # comma-separated integers or UUID strings

In [2]: collector = ResourceMetricCollector()                                   # log all devices and descendant processes of the current process on the GPUs
In [3]: collector = ResourceMetricCollector(root_pids={1})                      # log all devices and all GPU processes
In [4]: collector = ResourceMetricCollector(devices=Device(0), root_pids={1})   # log <GPU 0> and all GPU processes on <GPU 0>
In [5]: collector = ResourceMetricCollector(devices=Device.cuda.all())          # use the CUDA ordinal

In [6]: with collector(tag='<tag>'):
   ...:     # Do something
   ...:     collector.collect()  # -> Dict[str, float]
# key -> '<tag>/<scope>/<metric (unit)>/<mean/min/max>'
{
    '<tag>/host/cpu_percent (%)/mean': 8.967849777683456,
    '<tag>/host/cpu_percent (%)/min': 6.1,
    '<tag>/host/cpu_percent (%)/max': 28.1,
    ...,
    '<tag>/host/memory_percent (%)/mean': 21.5,
    '<tag>/host/swap_percent (%)/mean': 0.3,
    '<tag>/host/memory_used (GiB)/mean': 91.0136418208109,
    '<tag>/host/load_average (%) (1 min)/mean': 10.251427386878328,
    '<tag>/host/load_average (%) (5 min)/mean': 10.072539414569503,
    '<tag>/host/load_average (%) (15 min)/mean': 11.91126970422139,
    ...,
    '<tag>/cuda:0 (gpu:3)/memory_used (MiB)/mean': 3.875,
    '<tag>/cuda:0 (gpu:3)/memory_free (MiB)/mean': 11015.562499999998,
    '<tag>/cuda:0 (gpu:3)/memory_total (MiB)/mean': 11019.437500000002,
    '<tag>/cuda:0 (gpu:3)/memory_percent (%)/mean': 0.0,
    '<tag>/cuda:0 (gpu:3)/gpu_utilization (%)/mean': 0.0,
    '<tag>/cuda:0 (gpu:3)/memory_utilization (%)/mean': 0.0,
    '<tag>/cuda:0 (gpu:3)/fan_speed (%)/mean': 22.0,
    '<tag>/cuda:0 (gpu:3)/temperature (C)/mean': 25.0,
    '<tag>/cuda:0 (gpu:3)/power_usage (W)/mean': 19.11166264116916,
    ...,
    '<tag>/cuda:1 (gpu:2)/memory_used (MiB)/mean': 8878.875,
    ...,
    '<tag>/cuda:2 (gpu:1)/memory_used (MiB)/mean': 8182.875,
    ...,
    '<tag>/cuda:3 (gpu:0)/memory_used (MiB)/mean': 9286.875,
    ...,
    '<tag>/pid:12345/host/cpu_percent (%)/mean': 151.34342772112265,
    '<tag>/pid:12345/host/host_memory (MiB)/mean': 44749.72373447514,
    '<tag>/pid:12345/host/host_memory_percent (%)/mean': 8.675082352111717,
    '<tag>/pid:12345/host/running_time (min)': 336.23803206741576,
    '<tag>/pid:12345/cuda:1 (gpu:4)/gpu_memory (MiB)/mean': 8861.0,
    '<tag>/pid:12345/cuda:1 (gpu:4)/gpu_memory_percent (%)/mean': 80.4,
    '<tag>/pid:12345/cuda:1 (gpu:4)/gpu_memory_utilization (%)/mean': 6.711118172407917,
    '<tag>/pid:12345/cuda:1 (gpu:4)/gpu_sm_utilization (%)/mean': 48.23283397736476,
    ...,
    '<tag>/duration (s)': 7.247399162035435,
    '<tag>/timestamp': 1655909466.9981883
}
```

The results can be easily logged into [TensorBoard](https://github.com/tensorflow/tensorboard) or to CSV file. For example:

```python
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from nvitop import CudaDevice, ResourceMetricCollector
from nvitop.callbacks.tensorboard import add_scalar_dict

# Build networks and prepare datasets
...

# Logger and status collector
writer = SummaryWriter()
collector = ResourceMetricCollector(devices=CudaDevice.all(),  # log all visible CUDA devices and use the CUDA ordinal
                                    root_pids={os.getpid()},   # only log the descendant processes of the current process
                                    interval=1.0)              # snapshot interval for background daemon thread

# Start training
global_step = 0
for epoch in range(num_epoch):
    with collector(tag='train'):
        for batch in train_dataset:
            with collector(tag='batch'):
                metrics = train(net, batch)
                global_step += 1
                add_scalar_dict(writer, 'train', metrics, global_step=global_step)
                add_scalar_dict(writer, 'resources',      # tag='resources/train/batch/...'
                                collector.collect(),
                                global_step=global_step)

        add_scalar_dict(writer, 'resources',              # tag='resources/train/...'
                        collector.collect(),
                        global_step=epoch)

    with collector(tag='validate'):
        metrics = validate(net, validation_dataset)
        add_scalar_dict(writer, 'validate', metrics, global_step=epoch)
        add_scalar_dict(writer, 'resources',              # tag='resources/validate/...'
                        collector.collect(),
                        global_step=epoch)
```

Another example for logging to CSV file:

```python
import datetime
import time

import pandas as pd

from nvitop import ResourceMetricCollector

collector = ResourceMetricCollector(root_pids={1}, interval=2.0)  # log all devices and all GPU processes
df = pd.DataFrame()

with collector(tag='resources'):
    for _ in range(60):
        # Do something
        time.sleep(60)

        metrics = collector.collect()
        df_metrics = pd.DataFrame.from_records(metrics, index=[len(df)])
        df = pd.concat([df, df_metrics], ignore_index=True)
        # Flush to CSV file ...

df.insert(0, 'time', df['resources/timestamp'].map(datetime.datetime.fromtimestamp))
df.to_csv('results.csv', index=False)
```

You can also daemonize the collector in background using [`collect_in_background`](https://nvitop.readthedocs.io/en/latest/core/collector.html#nvitop.collect_in_background) or [`ResourceMetricCollector.daemonize`](https://nvitop.readthedocs.io/en/latest/core/collector.html#nvitop.ResourceMetricCollector.daemonize) with callback functions.

```python
from nvitop import Device, ResourceMetricCollector, collect_in_background

logger = ...

def on_collect(metrics):  # will be called periodically
    if logger.is_closed():  # closed manually by user
        return False
    logger.log(metrics)
    return True

def on_stop(collector):  # will be called only once at stop
    if not logger.is_closed():
        logger.close()  # cleanup

# Record metrics to the logger in background every 5 seconds.
# It will collect 5-second mean/min/max for each metric.
collect_in_background(
    on_collect,
    ResourceMetricCollector(Device.cuda.all()),
    interval=5.0,
    on_stop=on_stop,
)
```

or simply:

```python
ResourceMetricCollector(Device.cuda.all()).daemonize(
    on_collect,
    interval=5.0,
    on_stop=on_stop,
)
```

------

#### Low-level APIs

The full API references can be found at <https://nvitop.readthedocs.io>.

##### Device

The [device module](https://nvitop.readthedocs.io/en/latest/core/device.html) provides:

<table class="autosummary longtable docutils align-default">
  <colgroup>
    <col style="width: 10%" />
    <col style="width: 90%" />
  </colgroup>
  <tbody>
    <tr class="row-odd">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/device.html#nvitop.Device" title="nvitop.Device"><code class="xref py py-obj docutils literal notranslate"><span class="pre">Device</span></code></a>([index, uuid, bus_id])</p></td>
      <td><p>Live class of the GPU devices, different from the device snapshots.</p></td>
    </tr>
    <tr class="row-even">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/device.html#nvitop.PhysicalDevice" title="nvitop.PhysicalDevice"><code class="xref py py-obj docutils literal notranslate"><span class="pre">PhysicalDevice</span></code></a>([index, uuid, bus_id])</p></td>
      <td><p>Class for physical devices.</p></td>
    </tr>
    <tr class="row-odd">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/device.html#nvitop.MigDevice" title="nvitop.MigDevice"><code class="xref py py-obj docutils literal notranslate"><span class="pre">MigDevice</span></code></a>([index, uuid, bus_id])</p></td>
      <td><p>Class for MIG devices.</p></td>
    </tr>
    <tr class="row-even">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/device.html#nvitop.CudaDevice" title="nvitop.CudaDevice"><code class="xref py py-obj docutils literal notranslate"><span class="pre">CudaDevice</span></code></a>([cuda_index, nvml_index, uuid])</p></td>
      <td><p>Class for devices enumerated over the CUDA ordinal.</p></td>
    </tr>
    <tr class="row-odd">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/device.html#nvitop.CudaMigDevice" title="nvitop.CudaMigDevice"><code class="xref py py-obj docutils literal notranslate"><span class="pre">CudaMigDevice</span></code></a>([cuda_index, nvml_index, uuid])</p></td>
      <td><p>Class for CUDA devices that are MIG devices.</p></td>
    </tr>
    <tr class="row-even">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/device.html#nvitop.parse_cuda_visible_devices" title="nvitop.parse_cuda_visible_devices"><code class="xref py py-obj docutils literal notranslate"><span class="pre">parse_cuda_visible_devices</span></code></a>([...])</p></td>
      <td><p>Parses the given <code class="docutils literal notranslate"><span class="pre">CUDA_VISIBLE_DEVICES</span></code> value into a list of NVML device indices.</p></td>
    </tr>
    <tr class="row-odd">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/device.html#nvitop.normalize_cuda_visible_devices" title="nvitop.normalize_cuda_visible_devices"><code class="xref py py-obj docutils literal notranslate"><span class="pre">normalize_cuda_visible_devices</span></code></a>([...])</p></td>
      <td><p>Parses the given <code class="docutils literal notranslate"><span class="pre">CUDA_VISIBLE_DEVICES</span></code> value and convert it into a comma-separated string of UUIDs.</p></td>
    </tr>
  </tbody>
</table>

```python
In [1]: from nvitop import (
   ...:     host,
   ...:     Device, PhysicalDevice, CudaDevice, parse_cuda_visible_devices, normalize_cuda_visible_devices
   ...:     HostProcess, GpuProcess,
   ...:     NA,
   ...: )
   ...: import os
   ...: os.environ['CUDA_VISIBLE_DEVICES'] = '9,8,7,6'  # comma-separated integers or UUID strings

In [2]: Device.driver_version()
Out[2]: '430.64'

In [3]: Device.cuda_driver_version()  # the maximum CUDA version supported by the driver (can be different from the CUDA runtime version)
Out[3]: '10.1'

In [4]: Device.count()
Out[4]: 10

In [5]: CudaDevice.count()  # or `Device.cuda.count()`
Out[5]: 4

In [6]: all_devices      = Device.all()                 # all devices on board (physical device)
   ...: nvidia0, nvidia1 = Device.from_indices([0, 1])  # from physical device indices
   ...: all_devices
Out[6]: [
    PhysicalDevice(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=2, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=3, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=4, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=5, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=6, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=7, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=8, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    PhysicalDevice(index=9, name="GeForce RTX 2080 Ti", total_memory=11019MiB)
]

In [7]: # NOTE: The function results might be different between calls when the `CUDA_VISIBLE_DEVICES` environment variable has been modified
   ...: cuda_visible_devices = Device.from_cuda_visible_devices()  # from the `CUDA_VISIBLE_DEVICES` environment variable
   ...: cuda0, cuda1         = Device.from_cuda_indices([0, 1])    # from CUDA device indices (might be different from physical device indices if `CUDA_VISIBLE_DEVICES` is set)
   ...: cuda_visible_devices = CudaDevice.all()                    # shortcut to `Device.from_cuda_visible_devices()`
   ...: cuda_visible_devices = Device.cuda.all()                   # `Device.cuda` is aliased to `CudaDevice`
   ...: cuda_visible_devices
Out[7]: [
    CudaDevice(cuda_index=0, nvml_index=9, name="NVIDIA GeForce RTX 2080 Ti", total_memory=11019MiB),
    CudaDevice(cuda_index=1, nvml_index=8, name="NVIDIA GeForce RTX 2080 Ti", total_memory=11019MiB),
    CudaDevice(cuda_index=2, nvml_index=7, name="NVIDIA GeForce RTX 2080 Ti", total_memory=11019MiB),
    CudaDevice(cuda_index=3, nvml_index=6, name="NVIDIA GeForce RTX 2080 Ti", total_memory=11019MiB)
]

In [8]: nvidia0 = Device(0)  # from device index (or `Device(index=0)`)
   ...: nvidia0
Out[8]: PhysicalDevice(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB)

In [9]: nvidia1 = Device(uuid='GPU-01234567-89ab-cdef-0123-456789abcdef')  # from UUID string (or just`Device('GPU-xxxxxxxx-...')`)
   ...: nvidia2 = Device(bus_id='00000000:06:00.0')                        # from PCI bus ID
   ...: nvidia1
Out[9]: PhysicalDevice(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB)

In [10]: cuda0 = CudaDevice(0)                        # from CUDA device index (equivalent to `CudaDevice(cuda_index=0)`)
    ...: cuda1 = CudaDevice(nvml_index=8)             # from physical device index
    ...: cuda3 = CudaDevice(uuid='GPU-xxxxxxxx-...')  # from UUID string
    ...: cuda4 = Device.cuda(4)                       # `Device.cuda` is aliased to `CudaDevice`
    ...: cuda0
Out[10]:
CudaDevice(cuda_index=0, nvml_index=9, name="NVIDIA GeForce RTX 2080 Ti", total_memory=11019MiB)

In [11]: nvidia0.memory_used()  # in bytes
Out[11]: 9293398016

In [12]: nvidia0.memory_used_human()
Out[12]: '8862MiB'

In [13]: nvidia0.gpu_utilization()  # in percentage
Out[13]: 5

In [14]: nvidia0.processes()  # type: Dict[int, GpuProcess]
Out[14]: {
    52059: GpuProcess(pid=52059, gpu_memory=7885MiB, type=C, device=PhysicalDevice(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=52059, name='ipython3', status='sleeping', started='14:31:22')),
    53002: GpuProcess(pid=53002, gpu_memory=967MiB, type=C, device=PhysicalDevice(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=53002, name='python', status='running', started='14:31:59'))
}

In [15]: nvidia1_snapshot = nvidia1.as_snapshot()
    ...: nvidia1_snapshot
Out[15]: PhysicalDeviceSnapshot(
    real=PhysicalDevice(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    bus_id='00000000:05:00.0',
    compute_mode='Default',
    clock_infos=ClockInfos(graphics=1815, sm=1815, memory=6800, video=1680),  # in MHz
    clock_speed_infos=ClockSpeedInfos(current=ClockInfos(graphics=1815, sm=1815, memory=6800, video=1680), max=ClockInfos(graphics=2100, sm=2100, memory=7000, video=1950)),  # in MHz
    current_driver_model='N/A',
    decoder_utilization=0,              # in percentage
    display_active='Disabled',
    display_mode='Disabled',
    encoder_utilization=0,              # in percentage
    fan_speed=22,                       # in percentage
    gpu_utilization=17,                 # in percentage (NOTE: this is the utilization rate of SMs, i.e. GPU percent)
    index=1,
    max_clock_infos=ClockInfos(graphics=2100, sm=2100, memory=7000, video=1950),  # in MHz
    memory_clock=6800,                  # in MHz
    memory_free=10462232576,            # in bytes
    memory_free_human='9977MiB',
    memory_info=MemoryInfo(total=11554717696, free=10462232576, used=1092485120)  # in bytes
    memory_percent=9.5,                 # in percentage (NOTE: this is the percentage of used GPU memory)
    memory_total=11554717696,           # in bytes
    memory_total_human='11019MiB',
    memory_usage='1041MiB / 11019MiB',
    memory_used=1092485120,             # in bytes
    memory_used_human='1041MiB',
    memory_utilization=7,               # in percentage (NOTE: this is the utilization rate of GPU memory bandwidth)
    mig_mode='N/A',
    name='GeForce RTX 2080 Ti',
    performance_state='P2',
    persistence_mode='Disabled',
    power_limit=250000,                 # in milliwatts (mW)
    power_status='66W / 250W',          # in watts (W)
    power_usage=66051,                  # in milliwatts (mW)
    sm_clock=1815,                      # in MHz
    temperature=39,                     # in Celsius
    total_volatile_uncorrected_ecc_errors='N/A',
    utilization_rates=UtilizationRates(gpu=17, memory=7, encoder=0, decoder=0),  # in percentage
    uuid='GPU-01234567-89ab-cdef-0123-456789abcdef'
)

In [16]: nvidia1_snapshot.memory_percent  # snapshot uses properties instead of function calls
Out[16]: 9.5

In [17]: nvidia1_snapshot['memory_info']  # snapshot also supports `__getitem__` by string
Out[17]: MemoryInfo(total=11554717696, free=10462232576, used=1092485120)

In [18]: nvidia1_snapshot.bar1_memory_info  # snapshot will automatically retrieve not presented attributes from `real`
Out[18]: MemoryInfo(total=268435456, free=257622016, used=10813440)
```

**NOTE:** Some entry values may be `'N/A'` (type: [`NaType`](https://nvitop.readthedocs.io/en/latest/index.html#nvitop.NaType), subclass of `str`) when the corresponding resources are not applicable. The [`NA`](https://nvitop.readthedocs.io/en/latest/index.html#nvitop.NA) value supports arithmetic operations. It acts like `math.nan: float`.

```python
>>> from nvitop import NA
>>> NA
'N/A'

>>> 'memory usage: {}'.format(NA)  # NA is an instance of `str`
'memory usage: N/A'
>>> NA.lower()                     # NA is an instance of `str`
'n/a'
>>> NA.ljust(5)                    # NA is an instance of `str`
'N/A  '
>>> NA + 'str'                     # string contamination if the operand is a string
'N/Astr'

>>> float(NA)                      # explicit conversion to float (`math.nan`)
nan
>>> NA + 1                         # auto-casting to float if the operand is a number
nan
>>> NA * 1024                      # auto-casting to float if the operand is a number
nan
>>> NA / (1024 * 1024)             # auto-casting to float if the operand is a number
nan
```

You can use `entry != 'N/A'` conditions to avoid exceptions. It's safe to use `float(entry)` for numbers while `NaType` will be converted to `math.nan`. For example:

```python
memory_used: Union[int, NaType] = device.memory_used()            # memory usage in bytes or `'N/A'`
memory_used_in_mib: float       = float(memory_used) / (1 << 20)  # memory usage in Mebibytes (MiB) or `math.nan`
```

It's safe to compare `NaType` with numbers, but `NaType` is always larger than any number:

```python
devices_by_used_memory = sorted(Device.all(), key=Device.memory_used, reverse=True)  # it's safe to compare `'N/A'` with numbers
devices_by_free_memory = sorted(Device.all(), key=Device.memory_free, reverse=True)  # please add `memory_free != 'N/A'` checks if sort in descending order here
```

See [`nvitop.NaType`](https://nvitop.readthedocs.io/en/latest/apis/index.html#nvitop.NaType) documentation for more details.

##### Process

The [process module](https://nvitop.readthedocs.io/en/latest/core/process.html) provides:

<table class="autosummary longtable docutils align-default">
  <colgroup>
    <col style="width: 10%" />
    <col style="width: 90%" />
  </colgroup>
  <tbody>
    <tr class="row-odd">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/process.html#nvitop.HostProcess" title="nvitop.HostProcess"><code class="xref py py-obj docutils literal notranslate"><span class="pre">HostProcess</span></code></a>([pid])</p></td>
      <td><p>Represents an OS process with the given PID.</p></td>
    </tr>
    <tr class="row-even">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/process.html#nvitop.GpuProcess" title="nvitop.GpuProcess"><code class="xref py py-obj docutils literal notranslate"><span class="pre">GpuProcess</span></code></a>(pid, device[, gpu_memory, ...])</p></td>
      <td><p>Represents a process with the given PID running on the given GPU device.</p></td>
    </tr>
    <tr class="row-odd">
      <td><p><a href="https://nvitop.readthedocs.io/en/latest/core/process.html#nvitop.command_join" title="nvitop.command_join"><code class="xref py py-obj docutils literal notranslate"><span class="pre">command_join</span></code></a>(cmdline)</p></td>
      <td><p>Returns a shell-escaped string from command line arguments.</p></td>
    </tr>
  </tbody>
</table>

```python
In [19]: processes = nvidia1.processes()  # type: Dict[int, GpuProcess]
    ...: processes
Out[19]: {
    23266: GpuProcess(pid=23266, gpu_memory=1031MiB, type=C, device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40'))
}

In [20]: process = processes[23266]
    ...: process
Out[20]: GpuProcess(pid=23266, gpu_memory=1031MiB, type=C, device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40'))

In [21]: process.status()  # GpuProcess will automatically inherit attributes from GpuProcess.host
Out[21]: 'running'

In [22]: process.cmdline()  # type: List[str]
Out[22]: ['python3', 'rllib_train.py']

In [23]: process.command()  # type: str
Out[23]: 'python3 rllib_train.py'

In [24]: process.cwd()  # GpuProcess will automatically inherit attributes from GpuProcess.host
Out[24]: '/home/xxxxxx/Projects/xxxxxx'

In [25]: process.gpu_memory_human()
Out[25]: '1031MiB'

In [26]: process.as_snapshot()
Out[26]: GpuProcessSnapshot(
    real=GpuProcess(pid=23266, gpu_memory=1031MiB, type=C, device=PhysicalDevice(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40')),
    cmdline=['python3', 'rllib_train.py'],
    command='python3 rllib_train.py',
    compute_instance_id='N/A',
    cpu_percent=98.5,                       # in percentage
    device=PhysicalDevice(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    gpu_encoder_utilization=0,              # in percentage
    gpu_decoder_utilization=0,              # in percentage
    gpu_instance_id='N/A',
    gpu_memory=1081081856,                  # in bytes
    gpu_memory_human='1031MiB',
    gpu_memory_percent=9.4,                 # in percentage (NOTE: this is the percentage of used GPU memory)
    gpu_memory_utilization=5,               # in percentage (NOTE: this is the utilization rate of GPU memory bandwidth)
    gpu_sm_utilization=0,                   # in percentage (NOTE: this is the utilization rate of SMs, i.e. GPU percent)
    host=HostProcessSnapshot(
        real=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40'),
        cmdline=['python3', 'rllib_train.py'],
        command='python3 rllib_train.py',
        cpu_percent=98.5,                   # in percentage
        host_memory=9113627439,             # in bytes
        host_memory_human='8691MiB',
        is_running=True,
        memory_percent=1.6849018430285683,  # in percentage
        name='python3',
        running_time=datetime.timedelta(days=1, seconds=80013, microseconds=470024),
        running_time_human='46:13:33',
        running_time_in_seconds=166413.470024,
        status='running',
        username='panxuehai'
    ),
    host_memory=9113627439,                 # in bytes
    host_memory_human='8691MiB',
    is_running=True,
    memory_percent=1.6849018430285683,      # in percentage (NOTE: this is the percentage of used host memory)
    name='python3',
    pid=23266,
    running_time=datetime.timedelta(days=1, seconds=80013, microseconds=470024),
    running_time_human='46:13:33',
    running_time_in_seconds=166413.470024,
    status='running',
    type='C',                               # 'C' for Compute / 'G' for Graphics / 'C+G' for Both
    username='panxuehai'
)

In [27]: process.uids()  # GpuProcess will automatically inherit attributes from GpuProcess.host
Out[27]: puids(real=1001, effective=1001, saved=1001)

In [28]: process.kill()  # GpuProcess will automatically inherit attributes from GpuProcess.host

In [29]: list(map(Device.processes, all_devices))  # all processes
Out[29]: [
    {
        52059: GpuProcess(pid=52059, gpu_memory=7885MiB, type=C, device=PhysicalDevice(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=52059, name='ipython3', status='sleeping', started='14:31:22')),
        53002: GpuProcess(pid=53002, gpu_memory=967MiB, type=C, device=PhysicalDevice(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=53002, name='python', status='running', started='14:31:59'))
    },
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {
        84748: GpuProcess(pid=84748, gpu_memory=8975MiB, type=C, device=PhysicalDevice(index=8, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=84748, name='python', status='running', started='11:13:38'))
    },
    {
        84748: GpuProcess(pid=84748, gpu_memory=8341MiB, type=C, device=PhysicalDevice(index=9, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=84748, name='python', status='running', started='11:13:38'))
    }
]

In [30]: this = HostProcess(os.getpid())
    ...: this
Out[30]: HostProcess(pid=35783, name='python', status='running', started='19:19:00')

In [31]: this.cmdline()  # type: List[str]
Out[31]: ['python', '-c', 'import IPython; IPython.terminal.ipapp.launch_new_instance()']

In [32]: this.command()  # not simply `' '.join(cmdline)` but quotes are added
Out[32]: 'python -c "import IPython; IPython.terminal.ipapp.launch_new_instance()"'

In [33]: this.memory_info()
Out[33]: pmem(rss=83988480, vms=343543808, shared=12079104, text=8192, lib=0, data=297435136, dirty=0)

In [34]: import cupy as cp
    ...: x = cp.zeros((10000, 1000))
    ...: this = GpuProcess(os.getpid(), cuda0)  # construct from `GpuProcess(pid, device)` explicitly rather than calling `device.processes()`
    ...: this
Out[34]: GpuProcess(pid=35783, gpu_memory=N/A, type=N/A, device=CudaDevice(cuda_index=0, nvml_index=9, name="NVIDIA GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=35783, name='python', status='running', started='19:19:00'))

In [35]: this.update_gpu_status()  # update used GPU memory from new driver queries
Out[35]: 267386880

In [36]: this
Out[36]: GpuProcess(pid=35783, gpu_memory=255MiB, type=C, device=CudaDevice(cuda_index=0, nvml_index=9, name="NVIDIA GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=35783, name='python', status='running', started='19:19:00'))

In [37]: id(this) == id(GpuProcess(os.getpid(), cuda0))  # IMPORTANT: the instance will be reused while the process is running
Out[37]: True
```

##### Host (inherited from [psutil](https://github.com/giampaolo/psutil))

```python
In [38]: host.cpu_count()
Out[38]: 88

In [39]: host.cpu_percent()
Out[39]: 18.5

In [40]: host.cpu_times()
Out[40]: scputimes(user=2346377.62, nice=53321.44, system=579177.52, idle=10323719.85, iowait=28750.22, irq=0.0, softirq=11566.87, steal=0.0, guest=0.0, guest_nice=0.0)

In [41]: host.load_average()
Out[41]: (14.88, 17.8, 19.91)

In [42]: host.virtual_memory()
Out[42]: svmem(total=270352478208, available=192275968000, percent=28.9, used=53350518784, free=88924037120, active=125081112576, inactive=44803993600, buffers=37006450688, cached=91071471616, shared=23820632064, slab=8200687616)

In [43]: host.memory_percent()
Out[43]: 28.9

In [44]: host.swap_memory()
Out[44]: sswap(total=65534947328, used=475136, free=65534472192, percent=0.0, sin=2404139008, sout=4259434496)

In [45]: host.swap_percent()
Out[45]: 0.0
```

------

## Screenshots

![Screen Recording](https://user-images.githubusercontent.com/16078332/113173772-508dc380-927c-11eb-84c5-b6f496e54c08.gif)

Example output of `nvitop -1`:

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/117765250-41793880-b260-11eb-8a1b-9c32868a46d4.png" alt="Screenshot">
</p>

Example output of `nvitop`:

<table>
  <tr valign="center" align="center">
    <td>Full</td>
    <td>Compact</td>
  </tr>
  <tr valign="top" align="center">
    <td><img src="https://user-images.githubusercontent.com/16078332/117765260-4342fc00-b260-11eb-9198-7bcfdd1db113.png" alt="Full"></td>
    <td><img src="https://user-images.githubusercontent.com/16078332/117765274-476f1980-b260-11eb-9afd-877cca54e0bc.png" alt="Compact"></td>
  </tr>
</table>

Tree-view screen (shortcut: <kbd>t</kbd>) for GPU processes and their ancestors:

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/123914889-7b3e0400-d9b2-11eb-9b71-a48971617c2a.png" alt="Tree-view">
</p>

**NOTE:** The process tree is built in backward (recursively back to the tree root). Only GPU processes along with their children and ancestors (parents and grandparents ...) will be shown. Not all running processes will be displayed.

Environment variable screen (shortcut: <kbd>e</kbd>):

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/123914881-7a0cd700-d9b2-11eb-8da1-26f7a3a7c2b6.png" alt="Environment Screen">
</p>

------

## License

`nvitop` is released under the **GNU General Public License, version 3 (GPLv3)**.

**NOTE:** Please feel free to use `nvitop` as a package or dependency for your own projects. However, if you want to add or modify some features of `nvitop`, or copy some source code of `nvitop` into your own code, the source code should also be released under the GPLv3 License (as `nvitop`  contains some modified source code from [ranger](https://github.com/ranger/ranger) under the GPLv3 License).
