# nvitop

![Python 3.5+](https://img.shields.io/badge/Python-3.5%2B-brightgreen.svg)
[![PyPI](https://img.shields.io/pypi/v/nvitop?label=PyPI)](https://pypi.org/project/nvitop)
![Status](https://img.shields.io/pypi/status/nvitop?label=Status)
![Downloads](https://img.shields.io/pypi/dm/nvitop?label=Downloads)
[![License](https://img.shields.io/github/license/XuehaiPan/nvitop?label=License)](#license)

An interactive NVIDIA-GPU process viewer, aimed to provide a one-stop solution for GPU process management. ([screenshots](#screenshots))

This project is inspired by [nvidia-htop](https://github.com/peci1/nvidia-htop) and [nvtop](https://github.com/Syllo/nvtop) for monitoring, and [gpustat](https://github.com/wookayin/gpustat) for integration.

[nvidia-htop](https://github.com/peci1/nvidia-htop) a tool for enriching the output of `nvidia-smi`. [nvidia-htop](https://github.com/peci1/nvidia-htop) uses regular expressions to read the output of `nvidia-smi` from a subprocess, which is inefficient. In the meanwhile, there is a powerful interactive GPU monitoring tool called [nvtop](https://github.com/Syllo/nvtop). But [nvtop](https://github.com/Syllo/nvtop) is written in *C*, which makes it lack of portability. And What is really inconvenient is that you should compile it yourself during installation. Therefore, I made this repo. I got a lot help when reading the source code of [ranger](https://github.com/ranger/ranger), the console file manager. Some files in this repo are copied and modified from [ranger](https://github.com/ranger/ranger) under the GPLv3 License.

This project is currently in the *beta phase*, and most features have been tested on Linux. If you are using Windows with CUDA-capable GPUs, please submit feedback on the issue page, thank you very much!

If this repo is useful to you, please star ⭐️ it to let more people know 🤗.

Compare to `nvidia-smi`:

![Screenshot Comparison](https://user-images.githubusercontent.com/16078332/117765245-3f16de80-b260-11eb-99c7-077cd5519074.png)

## Features

- **Informative and fancy output**: show more information than `nvidia-smi` with colorized fancy box drawing.
- **Monitor mode**: can run as a resource monitor, rather than print the results only once. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop), limited support with command `watch -c`)
- **Interactive**: responsive for user inputs in monitor mode. (vs. [gpustat](https://github.com/wookayin/gpustat) & [py3nvml](https://github.com/fbcotter/py3nvml))
- **Efficient**:
  - query device status using [*NVML Python bindings*](https://pypi.org/project/nvidia-ml-py) directly, instead of parsing the output of `nvidia-smi`. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop))
  - cache results with `ttl_cache` from [cachetools](https://github.com/tkem/cachetools). (vs. [gpustat](https://github.com/wookayin/gpustat))
  - display information using the `curses` library rather than `print` with ANSI escape codes. (vs. [py3nvml](https://github.com/fbcotter/py3nvml))
- **Portable**: work on both Linux and Windows.
  - get host process information using the cross-platform library [psutil](https://github.com/giampaolo/psutil) instead of calling `ps -p <pid>` in a subprocess. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop) & [py3nvml](https://github.com/fbcotter/py3nvml))
  - written in pure Python, easy to install with `pip`. (vs. [nvtop](https://github.com/Syllo/nvtop))
- **Integrable**: easy to integrate into other applications, more than monitoring. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop) & [nvtop](https://github.com/Syllo/nvtop))

## Requirements

- Python 3.5+
- NVIDIA Management Library (NVML)
- nvidia-ml-py
- psutil
- cachetools
- curses
- termcolor

**Note**: The [NVIDIA Management Library (*NVML*)](https://developer.nvidia.com/nvidia-management-library-nvml) is a C-based programmatic interface for monitoring and managing various states. The runtime version of NVML library ships with the NVIDIA display driver (available at [Download Drivers | NVIDIA](https://www.nvidia.com/Download/index.aspx)), or can be downloaded as part of the NVIDIA CUDA Toolkit (available at [CUDA Toolkit | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)). The lists of OS platforms and NVIDIA-GPUs supported by the NVML library can be found in the [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html).

## Installation

Install from PyPI ([https://pypi.org/project/nvitop](https://pypi.org/project/nvitop)):

```bash
$ pip3 install --upgrade nvitop
```

Install the latest version from GitHub (*recommended*):

```bash
$ pip3 install git+https://github.com/XuehaiPan/nvitop.git#egg=nvitop
```

Or, clone this repo and install manually:

```bash
$ git clone --depth=1 https://github.com/XuehaiPan/nvitop.git
$ cd nvitop
$ pip3 install .
```

## Usage

Query the device and process status. The output is similar to `nvidia-smi`, but has been enriched and colorized.

```bash
# Query status of all devices
$ nvitop

# Specify query devices
$ nvitop -o 0 1  # only show <GPU 0> and <GPU 1>

# Only show devices in `CUDA_VISIBLE_DEVICES`
$ nvitop -ov
```

*Note: `nvitop` uses only one character to indicate the type of processes. `C` stands for compute processes, `G` for graphics processes, and `X` for both (i.e. MI(X), in `nvidia-smi` it is `C+G`).*

Run as a resource monitor, like `htop`:

```bash
# Automatically configure the display mode according to the terminal size
$ nvitop -m

# Arbitrarily display as `full` mode
$ nvitop -m full

# Arbitrarily display as `compact` mode
$ nvitop -m compact

# Specify query devices
$ nvitop -m -o 0 1  # only show <GPU 0> and <GPU 1>

# Only show devices in `CUDA_VISIBLE_DEVICES`
$ nvitop -m -ov
```

Press `q` to return to the terminal.

For Docker users:

```bash
$ docker build -t nvitop:latest .
$ docker run -it --rm --gpus all --pid=host nvitop:latest -m
```

Type `nvitop --help` for more information:

```
usage: nvitop [-h] [-m [{auto,full,compact}]] [-o idx [idx ...]] [-ov]
              [--gpu-util-thresh th1 th2] [--mem-util-thresh th1 th2]
              [--ascii]

A interactive NVIDIA-GPU process viewer.

optional arguments:
  -h, --help            show this help message and exit
  -m [{auto,full,compact}], --monitor [{auto,full,compact}]
                        Run as a resource monitor. Continuously report query data,
                        rather than the default of just once.
                        If no argument is given, the default mode `auto` is used.
  -o idx [idx ...], --only idx [idx ...]
                        Only show the specified devices, suppress option `-ov`.
  -ov, --only-visible   Only show devices in environment variable `CUDA_VISIBLE_DEVICES`.
  --gpu-util-thresh th1 th2
                        Thresholds of GPU utilization to distinguish load intensity.
                        Coloring rules: light < th1 % <= moderate < th2 % <= heavy.
                        ( 1 <= th1 < th2 <= 99, defaults: 10 75 )
  --mem-util-thresh th1 th2
                        Thresholds of GPU memory utilization to distinguish load intensity.
                        Coloring rules: light < th1 % <= moderate < th2 % <= heavy.
                        ( 1 <= th1 < th2 <= 99, defaults: 10 80 )
  --ascii               Use ASCII characters only.
                        This option is useful for terminals that do not support Unicode symbols.
```

#### Keybindings for monitor mode

|                                                       Key | Binding                                                               |
| --------------------------------------------------------: | :-------------------------------------------------------------------- |
|                                                       `q` | Quit and return to the terminal.                                      |
|                                           `a` / `f` / `c` | Change the display mode to *auto* / *full* / *compact*.               |
|      `<Left>` / `<Right>` <br> `[` / `]` <br> `<S-Wheel>` | Scroll the host information of processes.                             |
|                            `<Home>` <br> `<C-a>` <br> `^` | Scroll the host information of processes to the beginning of line.    |
|                             `<End>` <br> `<C-e>` <br> `$` | Scroll the host information of selected processes to the end of line. |
| `<Up>` / `<Down>` <br> `<Tab>` / `<S-Tab>` <br> `<Wheel>` | Select and highlight process.                                         |
|                                                   `<Esc>` | Clear selection.                                                      |
|                                                       `T` | Send `signal.SIGTERM` to the selected process (terminate).            |
|                                                       `K` | Send `signal.SIGKILL` to the selected process (kill).                 |
|                                          `I` <br> `<C-c>` | Send `signal.SIGINT` to the selected process (interrupt).             |

**Note**: Press the `CTRL` key to multiply the mouse wheel events by `5`.

## More than Monitoring

`nvitop` can be easily integrated into other applications like [gpustat](https://github.com/wookayin/gpustat).

### Device

```python
In [1]: from nvitop.core import host, Device, HostProcess, GpuProcess

In [2]: Device.driver_version()
Out[2]: '430.64'

In [3]: Device.cuda_version()
Out[3]: '10.1'

In [4]: Device.count()
Out[4]: 10

In [5]: all_devices = Device.all()
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

In [7]: nvidia0.memory_used()
Out[7]: 9293398016

In [8]: nvidia0.memory_used_human()
Out[8]: '8862MiB'

In [9]: nvidia0.gpu_utilization()  # in percentage
Out[9]: 5

In [10]: nvidia0.processes()
Out[10]: OrderedDict([
    (52059, GpuProcess(device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=7885MiB, host=HostProcess(pid=52059, name='ipython3', status='sleeping', started='14:31:22'))),
    (53002, GpuProcess(device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=967MiB, host=HostProcess(pid=53002, name='python', status='running', started='14:31:59')))
])

In [11]: nvidia1 = Device(bus_id='00000000:05:00.0')  # from PCI bus ID
    ...: nvidia1
Out[11]: Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB)

In [12]: nvidia1_snapshot = nvidia1.take_snapshot()
    ...: nvidia1_snapshot
Out[12]: Snapshot(real=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), bus_id='00000000:05:00.0', compute_mode='Default', display_active='Off', display_color='yellow', ecc_errors='N/A', fan_speed='22%', gpu_display_color='yellow', gpu_loading_intensity='moderate', gpu_utilization=15, gpu_utilization_string='15%', index=1, loading_intensity='moderate', memory_display_color='green', memory_loading_intensity='light', memory_total=11554717696, memory_usage='1041MiB / 11019MiB', memory_used=1092485120, memory_utilization=9, memory_utilization_string='9%', name='GeForce RTX 2080 Ti', performance_state='P2', persistence_mode='Off', power_limit=250000, power_state='66W / 250W', power_usage=66279, temperature='37C')

In [13]: nvidia1_snapshot.memory_utilization_string  # snapshots use properties instead of function calls
Out[13]: '9%'
```

### Process

```python
In [14]: processes = nvidia1.processes()
    ...: processes
Out[14]: OrderedDict([
    (23266, GpuProcess(device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=1031MiB, host=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40')))
])

In [15]: process = processes[23266]
    ...: process
Out[15]: GpuProcess(device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=1031MiB, host=HostProcess(pid=23266, name='python3', status='running', started='2021-05-10 21:02:40'))

In [16]: process.status()
Out[16]: 'running'

In [17]: process.cmdline()
Out[17]: ['python3', 'rllib_train.py']

In [18]: process.command()
Out[18]: 'python3 rllib_train.py'

In [19]: process.cwd()
Out[19]: '/home/xxxxxx/Projects/xxxxxx'

In [20]: process.gpu_memory_human()
Out[20]: '1031MiB'

In [21]: process_snapshot = process.take_snapshot()
    ...: process_snapshot
Out[21]: Snapshot(real=GpuProcess(device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=1031MiB, host=HostProcess(pid=23266, name='python3 rllib_t', status='running', started='2021-05-10 21:02:40')), cmdline=['python3', 'rllib_train.py'], command='python3 rllib_train.py', cpu_percent=94.5, cpu_percent_string='94.5', device=Device(index=1, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=1081081856, gpu_memory_human='1031MiB', identity=(23266, 1620651760.15, 1), is_running=True, memory_percent=1.707362595155753, memory_percent_string='1.7', name='python3 rllib_t', pid=23266, running_time=datetime.timedelta(days=1, seconds=63100, microseconds=60701), running_time_human='41:31:40', type='C', username='panxuehai')

In [22]: process.kill()

In [23]: list(map(Device.processes, all_devices))  # all processes
Out[23]: [
    OrderedDict([
        (52059, GpuProcess(device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=7885MiB, host=HostProcess(pid=52059, name='ipython3', status='sleeping', started='14:31:22'))),
        (53002, GpuProcess(device=Device(index=0, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=967MiB, host=HostProcess(pid=53002, name='python', status='running', started='14:31:59')))
    ]),
    OrderedDict(),
    OrderedDict(),
    OrderedDict(),
    OrderedDict(),
    OrderedDict(),
    OrderedDict(),
    OrderedDict(),
    OrderedDict([
        (84748, GpuProcess(device=Device(index=8, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=8975MiB, host=HostProcess(pid=84748, name='python', status='running', started='11:13:38')))
    ]),
    OrderedDict([
        (84748, GpuProcess(device=Device(index=9, name="GeForce RTX 2080 Ti", total_memory=11019MiB), gpu_memory=8341MiB, host=HostProcess(pid=84748, name='python', status='running', started='11:13:38')))
    ])
]
```

### Host (inherited from [psutil](https://github.com/giampaolo/psutil))

```python
In [24]: host.cpu_count()
Out[24]: 88

In [25]: host.cpu_percent()
Out[25]: 18.5

In [26]: host.cpu_times()
Out[26]: scputimes(user=2346377.62, nice=53321.44, system=579177.52, idle=10323719.85, iowait=28750.22, irq=0.0, softirq=11566.87, steal=0.0, guest=0.0, guest_nice=0.0)

In [27]: host.load_average()
Out[27]: (14.88, 17.8, 19.91)

In [28]: host.virtual_memory()
Out[28]: svmem(total=270352478208, available=192275968000, percent=28.9, used=53350518784, free=88924037120, active=125081112576, inactive=44803993600, buffers=37006450688, cached=91071471616, shared=23820632064, slab=8200687616)

In [29]: host.swap_memory()
Out[29]: sswap(total=65534947328, used=475136, free=65534472192, percent=0.0, sin=2404139008, sout=4259434496)
```

---

## Screenshots

![Screen Recording](https://user-images.githubusercontent.com/16078332/113173772-508dc380-927c-11eb-84c5-b6f496e54c08.gif)

Example output of `nvitop`:

<img src="https://user-images.githubusercontent.com/16078332/117765250-41793880-b260-11eb-8a1b-9c32868a46d4.png" alt="Screenshot">

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

## License

GNU General Public License, version 3 (GPLv3)
