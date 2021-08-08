# nvitop

![Python 3.5+](https://img.shields.io/badge/Python-3.5%2B-brightgreen.svg)
[![PyPI](https://img.shields.io/pypi/v/nvitop?label=PyPI)](https://pypi.org/project/nvitop)
![Status](https://img.shields.io/pypi/status/nvitop?label=Status)
[![Downloads](https://img.shields.io/pypi/dm/nvitop?label=Downloads)](https://pypistats.org/packages/nvitop)
[![License](https://img.shields.io/github/license/XuehaiPan/nvitop?label=License)](#license)

An interactive NVIDIA-GPU process viewer, the one-stop solution for GPU process management. ([screenshots](#screenshots))

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/117952038-5a104e00-b347-11eb-9ce5-27d2ac9fdd35.png" alt="Monitor">
  Monitor mode of <code>nvitop</code>.</br>(TERM: GNOME Terminal / OS: Ubuntu 16.04 LTS (over SSH) / Locale: <code>en_US.UTF-8</code>)
</p>

### Table of Contents  <!-- omit in toc -->

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Device and Process Status](#device-and-process-status)
  - [Resource Monitor](#resource-monitor)
    - [For Docker Users](#for-docker-users)
    - [For SSH Users](#for-ssh-users)
    - [Keybindings for Monitor Mode](#keybindings-for-monitor-mode)
  - [Callback Functions for Machine Learning Frameworks](#callback-functions-for-machine-learning-frameworks)
    - [Callback for TensorFlow (Keras)](#callback-for-tensorflow-keras)
    - [Callback for PyTorch Lightning](#callback-for-pytorch-lightning)
  - [More than Monitoring](#more-than-monitoring)
    - [Device](#device)
    - [Process](#process)
    - [Host (inherited from psutil)](#host-inherited-from-psutil)
- [Screenshots](#screenshots)
- [License](#license)
- [TODO List](#todo-list)

This project is inspired by [nvidia-htop](https://github.com/peci1/nvidia-htop) and [nvtop](https://github.com/Syllo/nvtop) for monitoring, and [gpustat](https://github.com/wookayin/gpustat) for application integration.

[nvidia-htop](https://github.com/peci1/nvidia-htop) is a tool for enriching the output of `nvidia-smi`. It uses regular expressions to read the output of `nvidia-smi` from a subprocess, which is inefficient. In the meanwhile, there is a powerful interactive GPU monitoring tool called [nvtop](https://github.com/Syllo/nvtop). But [nvtop](https://github.com/Syllo/nvtop) is written in *C*, which makes it lack of portability. And what is really inconvenient is that you should compile it yourself during the installation. Therefore, I made this repo. I got a lot help when reading the source code of [ranger](https://github.com/ranger/ranger), the console file manager. Some files in this repo are copied and modified from [ranger](https://github.com/ranger/ranger) under the **GPLv3 License**.

So far, `nvitop` is in the *beta phase*, and most features have been tested on Linux. If you are using Windows with NVIDIA-GPUs, please submit feedback on the issue page, thank you very much!

If this repo is useful to you, please star ⭐️ it to let more people know 🤗.

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/117765245-3f16de80-b260-11eb-99c7-077cd5519074.png" alt="Comparison">
  Compare to <code>nvidia-smi</code>.
</p>

## Features

- **Informative and fancy output**: show more information than `nvidia-smi` with colorized fancy box drawing.
- **Monitor mode**: can run as a resource monitor, rather than print the results only once. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop), limited support with command `watch -c`)
  - bar plots and history graphs
  - process sorting
  - process filtering
  - send signals to processes with a keystroke
  - tree-view screen for GPU processes and their parent processes
  - environment variable screen
  - help screen
- **Interactive**: responsive for user input in monitor mode. (vs. [gpustat](https://github.com/wookayin/gpustat) & [py3nvml](https://github.com/fbcotter/py3nvml))
- **Efficient**:
  - query device status using [*NVML Python bindings*](https://pypi.org/project/nvidia-ml-py) directly, instead of parsing the output of `nvidia-smi`. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop))
  - cache results with `ttl_cache` from [cachetools](https://github.com/tkem/cachetools). (vs. [gpustat](https://github.com/wookayin/gpustat))
  - display information using the `curses` library rather than `print` with ANSI escape codes. (vs. [py3nvml](https://github.com/fbcotter/py3nvml))
  - asynchronously gather information using multithreading and correspond to user input much faster. (vs. [nvtop](https://github.com/Syllo/nvtop))
- **Portable**: work on both Linux and Windows.
  - get host process information using the cross-platform library [psutil](https://github.com/giampaolo/psutil) instead of calling `ps -p <pid>` in a subprocess. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop) & [py3nvml](https://github.com/fbcotter/py3nvml))
  - written in pure Python, easy to install with `pip`. (vs. [nvtop](https://github.com/Syllo/nvtop))
- **Integrable**: easy to integrate into other applications, more than monitoring. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop) & [nvtop](https://github.com/Syllo/nvtop))

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/128634168-cad4bf87-770a-44ec-9da5-3496c3898706.png" alt="Windows">
  <code>nvitop</code> supports Windows!</br>
  (SHELL: PowerShell / TERM: Windows Terminal / OS: Windows 10 / Locale: <code>en-US</code>)
</p>

## Requirements

- Python 3.5+ (with `pip>=10.0`)
- NVIDIA Management Library (NVML)
- nvidia-ml-py
- psutil
- cachetools
- termcolor
- curses<sup>[*](#curses)</sup>

**NOTE:** The [NVIDIA Management Library (*NVML*)](https://developer.nvidia.com/nvidia-management-library-nvml) is a C-based programmatic interface for monitoring and managing various states. The runtime version of NVML library ships with the NVIDIA display driver (available at [Download Drivers | NVIDIA](https://www.nvidia.com/Download/index.aspx)), or can be downloaded as part of the NVIDIA CUDA Toolkit (available at [CUDA Toolkit | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)). The lists of OS platforms and NVIDIA-GPUs supported by the NVML library can be found in the [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html).

<a name="curses">*</a> The `curses` library is a built-in module of Python on Unix-like systems, and it is supported by the third-party package `windows-curses` on Windows.

## Installation

Install from PyPI ([![PyPI](https://img.shields.io/pypi/v/nvitop?label=PyPI)](https://pypi.org/project/nvitop) / ![Status](https://img.shields.io/pypi/status/nvitop?label=Status)):

```bash
pip3 install --upgrade nvitop
```

Install the latest version from GitHub (![Commit Count](https://img.shields.io/github/commits-since/XuehaiPan/nvitop/v0.3.6.2)):

```bash
pip3 install --force-reinstall git+https://github.com/XuehaiPan/nvitop.git#egg=nvitop
```

Or, clone this repo and install manually:

```bash
git clone --depth=1 https://github.com/XuehaiPan/nvitop.git
cd nvitop
pip3 install .
```

**IMPORTANT:** `pip` will install `nvidia-ml-py==11.450.51` as a dependency for `nvitop`. Please verify whether the `nvidia-ml-py` package is compatible with your NVIDIA driver version. You can check the release history of `nvidia-ml-py` at [nvidia-ml-py's Release History](https://pypi.org/project/nvidia-ml-py/11.450.51/#history), and install the compatible version manually by:

```bash
pip3 install --no-dependencies nvidia-ml-py==xx.yyy.zzz
```

Since `nvidia-ml-py>=11.450.129`, the definition of `nvmlProcessInfo_t` has introduced two new fields `gpuInstanceId` and `computeInstanceId` (`GI ID` and `CI ID` in newer `nvidia-smi`) which are incompatible with some old NVIDIA drivers. `nvitop` may not display the processes correctly due to this incompatibility.

## Usage

### Device and Process Status

Query the device and process status. The output is similar to `nvidia-smi`, but has been enriched and colorized.

```bash
# Query status of all devices
$ nvitop  # or use `python3 -m nvitop`

# Specify query devices (by integer indices)
$ nvitop -o 0 1  # only show <GPU 0> and <GPU 1>

# Only show devices in `CUDA_VISIBLE_DEVICES` (by integer indices or UUID strings)
$ nvitop -ov

# Only show GPU processes with the compute context (type: 'C' or 'C+G')
$ nvitop -c
```

The result will be displayed **ONLY ONCE**, which is consistent with the behavior of `nvidia-smi`. Type `nvitop --help` for more command options.

### Resource Monitor

Run as a resource monitor:

```bash
# Automatically configure the display mode according to the terminal size
$ nvitop -m  # or use `python3 -m nvitop -m`

# Arbitrarily display as `full` mode
$ nvitop -m full

# Arbitrarily display as `compact` mode
$ nvitop -m compact

# Specify query devices (by integer indices)
$ nvitop -m -o 0 1  # only show <GPU 0> and <GPU 1>

# Only show devices in `CUDA_VISIBLE_DEVICES` (by integer indices or UUID strings)
$ nvitop -m -ov

# Only show GPU processes with the compute context (type: 'C' or 'C+G')
$ nvitop -m -c

# Use ASCII characters only
$ nvitop -m -U  # useful for terminals without Unicode support
```

Press <kbd>h</kbd> for help or <kbd>q</kbd> to return to the terminal. See [Keybindings for Monitor Mode](#keybindings-for-monitor-mode) for more shortcuts.

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/128024364-7cb83ec3-1386-47f4-a61e-2f7ab44ceadc.png" alt="Help Screen">
  <code>nvitop</code> comes with a help screen (shortcut: <kbd>h</kbd>).
</p>

**HINT:** You can set the following alias in your shell profile to make `nvitop` always invoke the resource monitor:

```bash
alias nvitop='nvitop -m'                 # Bash / Zsh / Fish ...
function nvitop { nvitop.exe -m $args }  # PowerShell
```

In monitor mode, you can use <kbd>Ctrl-c</kbd> / <kbd>T</kbd> / <kbd>K</kbd> keys to interrupt / terminate / kill a process. And it's recommended to *terminate* or *kill* a process in the **tree-view screen** (shortcut: <kbd>t</kbd>). For normal users, `nvitop` will shallow other users' processes (in low-intensity colors). For **system administrators**, you can use `sudo nvitop -m` to terminate other users' processes.

#### For Docker Users

Build and run the Docker image using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker):

```bash
git clone --depth=1 https://github.com/XuehaiPan/nvitop.git && cd nvitop  # clone this repo first
docker build --tag nvitop:latest .  # build the Docker image
docker run --interactive --tty --rm --runtime=nvidia --gpus all --pid=host nvitop:latest -m  # run the Docker container
```

**NOTE:** Don't forget to add the `--pid=host` option when running the container.

#### For SSH Users

Run `nvitop` directly on the SSH session instead of a login shell:

```bash
ssh user@host -t nvitop -m                 # installed by `sudo pip3 install ...`
ssh user@host -t '~/.local/bin/nvitop' -m  # installed by `pip3 install --user ...`
```

**NOTE:** Users need to add the `-t` option to allocate a pseudo-terminal over the SSH session for monitor mode.

#### Keybindings for Monitor Mode

|                                                                        Key | Binding                                                                              |
| -------------------------------------------------------------------------: | :----------------------------------------------------------------------------------- |
|                                                                        `q` | Quit and return to the terminal.                                                     |
|                                                                  `h` / `?` | Go to the help screen.                                                               |
|                                                            `a` / `f` / `c` | Change the display mode to *auto* / *full* / *compact*.                              |
|                                                     `r` / `<C-r>` / `<F5>` | Force refresh the window.                                                            |
|                                                                            |                                                                                      |
|      `<Left>` / `<Right>`<br>`<A-h>` / `<A-l>`<br>`[` / `]`<br>`<S-Wheel>` | Scroll the host information of processes.                                            |
|                                                             `<C-a>`<br>`^` | Scroll left to the beginning of the process entry (i.e. beginning of line).          |
|                                                             `<C-e>`<br>`$` | Scroll right to the end of the process entry (i.e. end of line).                     |
| `<Up>` / `<Down>`<br>`<A-k>` / `<A-j>`<br>`<Tab>` / `<S-Tab>`<br>`<Wheel>` | Select and highlight a process.                                                      |
|                                                                   `<Home>` | Select the first process.                                                            |
|                                                                    `<End>` | Select the last process.                                                             |
|                                                                    `<Esc>` | Clear process selection.                                                             |
|                                                                        `e` | Show process environment.                                                            |
|                                                                        `t` | Toggle tree-view screen.                                                             |
|                                                                            |                                                                                      |
|                                                             `<C-c>`<br>`I` | Send `signal.SIGINT` to the selected process (interrupt).                            |
|                                                                        `T` | Send `signal.SIGTERM` to the selected process (terminate).                           |
|                                                                        `K` | Send `signal.SIGKILL` to the selected process (kill).                                |
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

Type `nvitop --help` for more command options:

```text
usage: nvitop [--help] [--version] [--monitor [{auto,full,compact}]] [--ascii]
              [--gpu-util-thresh th1 th2] [--mem-util-thresh th1 th2]
              [--only idx [idx ...]] [--only-visible] [--compute] [--graphics]
              [--user [USERNAME [USERNAME ...]]] [--pid PID [PID ...]]

An interactive NVIDIA-GPU process viewer.

optional arguments:
  --help, -h            show this help message and exit.
  --version, -V         show nvitop's version number and exit.
  --monitor [{auto,full,compact}], -m [{auto,full,compact}]
                        Run as a resource monitor. Continuously report query data,
                        rather than the default of just once.
                        If no argument is given, the default mode `auto` is used.
  --ascii, --no-unicode, -U
                        Use ASCII characters only, which is useful for terminals without Unicode support.
  --gpu-util-thresh th1 th2
                        Thresholds of GPU utilization to determine the load intensity.
                        Coloring rules: light < th1 % <= moderate < th2 % <= heavy.
                        ( 1 <= th1 < th2 <= 99, defaults: 10 75 )
  --mem-util-thresh th1 th2
                        Thresholds of GPU memory utilization to determine the load intensity.
                        Coloring rules: light < th1 % <= moderate < th2 % <= heavy.
                        ( 1 <= th1 < th2 <= 99, defaults: 10 80 )

device filtering:
  --only idx [idx ...], -o idx [idx ...]
                        Only show the specified devices, suppress option `--only-visible`.
  --only-visible, -ov   Only show devices in environment variable `CUDA_VISIBLE_DEVICES`.

process filtering:
  --compute, -c         Only show GPU processes with the compute context. (type: 'C' or 'C+G')
  --graphics, -g        Only show GPU processes with the graphics context. (type: 'G' or 'C+G')
  --user [USERNAME [USERNAME ...]], -u [USERNAME [USERNAME ...]]
                        Only show processes of the given users (or `$USER` for no argument).
  --pid PID [PID ...], -p PID [PID ...]
                        Only show processes of the given PIDs.
```

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
from nvitop.callbacks.lightning import GpuStatsLogger
gpu_stats = GpuStatsLogger()
trainer = Trainer(gpus=[..], logger=True, callbacks=[gpu_stats])
```

**NOTE:** Users should assign a logger to the trainer.

### More than Monitoring

`nvitop` can be easily integrated into other applications.

#### Device

```python
In [1]: from nvitop import host, Device, HostProcess, GpuProcess, NA

In [2]: Device.driver_version()
Out[2]: '430.64'

In [3]: Device.cuda_version()
Out[3]: '10.1'

In [4]: Device.count()
Out[4]: 10

In [5]: all_devices          = Device.all()                        # all devices on board
      : nvidia_0_1           = Device.from_indices([0, 1])         # from device indices
      : cuda_visible_devices = Device.from_cuda_visible_devices()  # from environment variable `CUDA_VISIBLE_DEVICES`
      : cuda_0_1             = Device.from_cuda_indices([0, 1])    # from CUDA device ID (might be different from device ID if `CUDA_VISIBLE_DEVICES` is set)
   ...: all_devices
Out[5]: [
    Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=2, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=3, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=4, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=5, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=6, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=7, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=8, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    Device(index=9, name="GeForce RTX 2080 Ti", total_memory=11019MiB)
]

In [6]: nvidia0 = Device(0)  # from device index
   ...: nvidia0
Out[6]: Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB)

In [7]: nvidia0.memory_used()  # in bytes
Out[7]: 9293398016

In [8]: nvidia0.memory_used_human()
Out[8]: '8862MiB'

In [9]: nvidia0.gpu_utilization()  # in percentage
Out[9]: 5

In [10]: nvidia0.processes()
Out[10]: {
    52059: GpuProcess(pid=52059, gpu_memory=7885MiB, type=C, device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=52059, name='ipython3', status='sleeping', started='14:31:22')),
    53002: GpuProcess(pid=53002, gpu_memory=967MiB, type=C, device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=53002, name='python', status='running', started='14:31:59'))
}

In [11]: nvidia1 = Device(bus_id='00000000:05:00.0')  # from PCI bus ID
    ...: nvidia1
Out[11]: Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB)

In [12]: nvidia1_snapshot = nvidia1.as_snapshot()
    ...: nvidia1_snapshot
Out[12]: DeviceSnapshot(
    real=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    bus_id='00000000:05:00.0',
    compute_mode='Default',
    display_active='Off',
    ecc_errors='N/A',
    fan_speed=22,                       # in percentage
    fan_speed_string='22%',             # in percentage
    gpu_utilization=17,                 # in percentage
    gpu_utilization_string='17%',       # in percentage
    index=1,
    memory_free=10462232576,            # in bytes
    memory_free_human='9977MiB',
    memory_total=11554717696,           # in bytes
    memory_total_human='11019MiB',
    memory_usage='1041MiB / 11019MiB',
    memory_used=1092485120,             # in bytes
    memory_used_human='1041MiB',
    memory_utilization=9.5,             # in percentage
    memory_utilization_string='9.5%',   # in percentage
    name='GeForce RTX 2080 Ti',
    performance_state='P2',
    persistence_mode='Off',
    power_limit=250000,                 # in milliwatts (mW)
    power_status='66W / 250W',          # in watts (W)
    power_usage=66051,                  # in milliwatts (mW)
    temperature=39,                     # in Celsius
    temperature_string='39C'            # in Celsius
)

In [13]: nvidia1_snapshot.memory_utilization_string  # snapshot uses properties instead of function calls
Out[13]: '9%'

In [14]: nvidia1_snapshot.encoder_utilization  # snapshot will automatically retrieve not presented attributes from `real`
Out[14]: [0, 1000000]

In [15]: nvidia1_snapshot
Out[15]: DeviceSnapshot(
    real=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    bus_id='00000000:05:00.0',
    compute_mode='Default',
    display_active='Off',
    ecc_errors='N/A',
    encoder_utilization=[0, 1000000],   ##### <-- new entry #####
    fan_speed=22,                       # in percentage
    fan_speed_string='22%',             # in percentage
    gpu_utilization=17,                 # in percentage
    gpu_utilization_string='17%',       # in percentage
    index=1,
    memory_free=10462232576,            # in bytes
    memory_free_human='9977MiB',
    memory_total=11554717696,           # in bytes
    memory_total_human='11019MiB',
    memory_usage='1041MiB / 11019MiB',
    memory_used=1092485120,             # in bytes
    memory_used_human='1041MiB',
    memory_utilization=9.5,             # in percentage
    memory_utilization_string='9.5%',   # in percentage
    name='GeForce RTX 2080 Ti',
    performance_state='P2',
    persistence_mode='Off',
    power_limit=250000,                 # in milliwatts (mW)
    power_status='66W / 250W',          # in watts (W)
    power_usage=66051,                  # in milliwatts (mW)
    temperature=39,                     # in Celsius
    temperature_string='39C'            # in Celsius
)
```

**NOTE:** Some entry values may be `'N/A'` (type: `NaType`, subclass of `str`) when the corresponding resources are not applicable. You can use `entry != 'N/A'` conditions to avoid exceptions. It's safe to use `float(entry)` for numbers while `NaType` will be converted to `math.nan`. For example:

```python
memory_used: Union[int, NaType] = device.memory_used()            # memory usage in bytes or `'N/A'`
memory_used_in_mib: float       = float(memory_used) / (1 << 20)  # memory usage in Mebibytes (MiB) or `math.nan`
```

It's safe to compare `NaType` with numbers, but `NaType` is always larger than any number:

```python
devices_by_used_memory = sorted(Device.all(), key=Device.memory_used, reverse=True)  # it's safe to compare `'N/A'` with numbers
devices_by_free_memory = sorted(Device.all(), key=Device.memory_free, reverse=True)  # please add `memory_free != 'N/A'` checks if sort in descending order here
```

#### Process

```python
In [16]: processes = nvidia1.processes()  # type: Dict[int, GpuProcess]
    ...: processes
Out[16]: {
    23266: GpuProcess(pid=23266, gpu_memory=1031MiB, type=C, device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40'))
}

In [17]: process = processes[23266]
    ...: process
Out[17]: GpuProcess(pid=23266, gpu_memory=1031MiB, type=C, device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40'))

In [18]: process.status()
Out[18]: 'running'

In [19]: process.cmdline()  # type: List[str]
Out[19]: ['python3', 'rllib_train.py']

In [20]: process.command()  # type: str
Out[20]: 'python3 rllib_train.py'

In [21]: process.cwd()
Out[21]: '/home/xxxxxx/Projects/xxxxxx'

In [22]: process.gpu_memory_human()
Out[22]: '1031MiB'

In [23]: process.as_snapshot()
Out[23]: GpuProcessSnapshot(
    real=GpuProcess(pid=23266, gpu_memory=1031MiB, type=C, device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40')),
    cmdline=['python3', 'rllib_train.py'],
    command='python3 rllib_train.py',
    cpu_percent=98.5,                      # in percentage
    cpu_percent_string='98.5%',            # in percentage
    device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB),
    gpu_encoder_utilization=0,             # in percentage
    gpu_encoder_utilization_string='0%',   # in percentage
    gpu_decoder_utilization=0,             # in percentage
    gpu_decoder_utilization_string='0%',   # in percentage
    gpu_memory=1081081856,                 # in bytes
    gpu_memory_human='1031MiB',
    gpu_memory_utilization=9.4,            # in percentage
    gpu_memory_utilization_string='9.4%',  # in percentage
    gpu_sm_utilization=0,                  # in percentage
    gpu_sm_utilization_string='0%',        # in percentage
    identity=(23266, 1620651760.15, 1),
    is_running=True,
    memory_percent=1.6849018430285683,     # in percentage
    memory_percent_string='1.7%',          # in percentage
    name='python3',
    pid=23266,
    running_time=datetime.timedelta(days=1, seconds=80013, microseconds=470024),
    running_time_human='46:13:33',
    type='C',                             # 'C' for Compute / 'G' for Graphics / 'C+G' for Both
    username='panxuehai'
)

In [24]: process.kill()

In [25]: list(map(Device.processes, all_devices))  # all processes
Out[25]: [
    {
        52059: GpuProcess(pid=52059, gpu_memory=7885MiB, type=C, device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=52059, name='ipython3', status='sleeping', started='14:31:22')),
        53002: GpuProcess(pid=53002, gpu_memory=967MiB, type=C, device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=53002, name='python', status='running', started='14:31:59'))
    },
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {
        84748: GpuProcess(pid=84748, gpu_memory=8975MiB, type=C, device=Device(index=8, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=84748, name='python', status='running', started='11:13:38'))
    },
    {
        84748: GpuProcess(pid=84748, gpu_memory=8341MiB, type=C, device=Device(index=9, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=84748, name='python', status='running', started='11:13:38'))
    }
]

In [26]: import os
    ...: this = HostProcess(os.getpid())
    ...: this
Out[26]: HostProcess(pid=35783, name='python', status='running', started='19:19:00')

In [27]: this.cmdline()  # type: List[str]
Out[27]: ['python', '-c', 'import IPython; IPython.terminal.ipapp.launch_new_instance()']

In [27]: this.command()  # not simply `' '.join(cmdline)` but quotes are added
Out[27]: 'python -c "import IPython; IPython.terminal.ipapp.launch_new_instance()"'

In [28]: this.memory_info()
Out[28]: pmem(rss=83988480, vms=343543808, shared=12079104, text=8192, lib=0, data=297435136, dirty=0)

In [29]: import cupy as cp
    ...: x = cp.zeros((10000, 1000))
    ...: this = GpuProcess(os.getpid(), nvidia0)  # construct from `GpuProcess(pid, device)` explicitly rather than calling `device.processes()`
    ...: this
Out[29]: GpuProcess(pid=35783, gpu_memory=N/A, type=N/A, device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=35783, name='python', status='running', started='19:19:00'))

In [30]: this.update_gpu_status()  # update used GPU memory from new driver queries
Out[30]: 267386880

In [31]: this
Out[31]: GpuProcess(pid=35783, gpu_memory=255MiB, type=C, device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), host=HostProcess(pid=35783, name='python', status='running', started='19:19:00'))

In [32]: id(this) == id(GpuProcess(os.getpid(), nvidia0))  # IMPORTANT: the instance will be reused while the process is running
Out[32]: True
```

#### Host (inherited from [psutil](https://github.com/giampaolo/psutil))

```python
In [33]: host.cpu_count()
Out[33]: 88

In [34]: host.cpu_percent()
Out[34]: 18.5

In [35]: host.cpu_times()
Out[35]: scputimes(user=2346377.62, nice=53321.44, system=579177.52, idle=10323719.85, iowait=28750.22, irq=0.0, softirq=11566.87, steal=0.0, guest=0.0, guest_nice=0.0)

In [36]: host.load_average()
Out[36]: (14.88, 17.8, 19.91)

In [37]: host.virtual_memory()
Out[37]: svmem(total=270352478208, available=192275968000, percent=28.9, used=53350518784, free=88924037120, active=125081112576, inactive=44803993600, buffers=37006450688, cached=91071471616, shared=23820632064, slab=8200687616)

In [38]: host.swap_memory()
Out[38]: sswap(total=65534947328, used=475136, free=65534472192, percent=0.0, sin=2404139008, sout=4259434496)
```

---

## Screenshots

![Screen Recording](https://user-images.githubusercontent.com/16078332/113173772-508dc380-927c-11eb-84c5-b6f496e54c08.gif)

Example output of `nvitop`:

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/117765250-41793880-b260-11eb-8a1b-9c32868a46d4.png" alt="Screenshot">
</p>

Example output of `nvitop -m`:

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

Tree-view screen (shortcut: <kbd>t</kbd>) for GPU processes and their parents:

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/123914889-7b3e0400-d9b2-11eb-9b71-a48971617c2a.png" alt="Tree-view">
</p>

**NOTE:** The process tree is built in backward (recursively back to the tree root). Only GPU processes and their parents (and grandparents ...) will be shown. Not all child processes will be displayed.

Environment variable screen (shortcut: <kbd>e</kbd>):

<p align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/16078332/123914881-7a0cd700-d9b2-11eb-8da1-26f7a3a7c2b6.png" alt="Environment Screen">
</p>

## License

`nvitop` is released under the **GNU General Public License, version 3 (GPLv3)**.

**NOTE:** Please feel free to use `nvitop` as a package or dependency for your own projects. However, if you want to add or modify some features of `nvitop`, or copy some source code of `nvitop` into your own code, the source code should also be released under the GPLv3 License (as `nvitop`  contains some modified source code from [ranger](https://github.com/ranger/ranger) under the GPLv3 License).

## TODO List

- [X] colorize device information based on the load intensity
- [X] basic process information both on the device and host
- [X] GPU process management (interrupt / terminate / kill)
- [X] bar plots and history graphs
- [X] process sorting
- [X] help screen
- [X] callbacks for [TensorFlow (Keras)](https://www.tensorflow.org) and [PyTorch Lightning](https://pytorchlightning.ai)
- [X] process environment variable screen
- [X] process filtering
- [X] process management for parent processes (tree view / interrupt / terminate / kill)
- [ ] scrollable process list for large amounts of processes
- [ ] web interface (under consideration)
- [ ] AMD ROCm support (help wanted for testing)
