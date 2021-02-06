# nvhtop

An interactive Nvidia-GPU process viewer.

This project is inspired by [`nvidia-htop`](https://github.com/peci1/nvidia-htop), a tool for enriching the output of `nvidia-smi`. [`nvidia-htop`](https://github.com/peci1/nvidia-htop) uses regular expressions to read the output of `nvidia-smi` from a subprocess, which is inefficient. Meanwhile, there is a powerful interactive GPU monitoring tool called [`nvtop`](https://github.com/Syllo/nvtop). But [`nvtop`](https://github.com/Syllo/nvtop) is written in C, which makes it lack of portability. And you should compile it yourself during installation, which is really inconvenient. Therefore, based on the above reasons, I make this repo. I got a lot help when reading the source code of [`ranger`](https://github.com/ranger/ranger), the console file manager. Some files in this repo are copied and modified from [`ranger`](https://github.com/ranger/ranger) under the GPLv3 License.

## Features

- **Informative and fancy output**: show more information than `nvidia-smi` with colorized fancy box drawing.
- **Monitor mode**: can run as a resource monitor, rather than print the results only once. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop))
- **Interactive**: responsive for user inputs in monitor mode. (vs. [py3nvml](https://github.com/fbcotter/py3nvml))
- **Efficiency**:
  - query status using NVML Python bindings directly and cache them with `ttl_cache`. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop))
  - display information using `curses` library rather than `print` with ANSI escape codes. (vs. [py3nvml](https://github.com/fbcotter/py3nvml))
- **Portability**:
  - get process information using `psutil` library instead of calling `ps -p`, and works on both Linux and Windows. (vs. [nvidia-htop](https://github.com/peci1/nvidia-htop))
  - written in pure Python, easy to install, and works on both Linux and Windows. (vs. [nvtop](https://github.com/Syllo/nvtop))

## Requirements

- Python 3.5+
- curses
- nvidia-ml-py
- psutil
- cachetools
- termcolor

## Installation

Install `nvhtop` using `pip` from GitHub:

```bash
$ pip install git+https://github.com/XuehaiPan/nvhtop.git#egg=nvhtop
```

Or, clone this repo and install `nvhtop` manually:

```bash
$ git clone --depth=1 https://github.com/XuehaiPan/nvhtop.git
$ cd nvhtop
$ pip install .
```

## Usage

Query the device and process status. The output is similar to `nvidia-smi`, but has been enriched and colorized.

```bash
$ nvhtop
```

Run as a resource monitor, like `htop`:

```bash
# Automatically configure the display mode according to the terminal size
$ nvhtop --monitor
$ nvhtop --monitor auto

# Forcibly display as `full` mode
$ nvhtop --monitor full

# Forcibly display as `compact` mode
$ nvhtop --monitor compact
```

Press `q` to return to the terminal.

Type `nvhtop --help` for more information:

```
usage: nvhtop [-h] [-m [{auto,full,compact}]] [--gpu-util-thresh th1 th2]
              [--mem-util-thresh th1 th2]

A interactive Nvidia-GPU process viewer.

optional arguments:
  -h, --help            show this help message and exit
  -m [{auto,full,compact}], --monitor [{auto,full,compact}]
                        Run as a resource monitor. Continuously report query data,
                        rather than the default of just once.
                        If no argument is specified, the default mode `auto` is used.
  --gpu-util-thresh th1 th2
                        Thresholds of GPU utilization to distinguish load intensity.
                        Coloring rules: light < th1 % <= moderate < th2 % <= heavy.
                        ( 1 <= th1 < th2 <= 99, defaults: 10 75 )
  --mem-util-thresh th1 th2
                        Thresholds of GPU memory utilization to distinguish load intensity.
                        Coloring rules: light < th1 % <= moderate < th2 % <= heavy.
                        ( 1 <= th1 < th2 <= 99, defaults: 10 80 )
```

## Screenshots

Example output of `nvhtop`:

<img width="680" alt="Screenshot" src="https://user-images.githubusercontent.com/16078332/107119517-0a8f3180-68c3-11eb-9569-2274c17e2c5f.png">

Example output of `nvhtop --monitor`:

<table>
  <tr valign="center">
    <td align="center">Full</td>
    <td align="center">Compact</td>
  </tr>
  <tr valign="top">
    <td align="center"><img src="https://user-images.githubusercontent.com/16078332/107119519-0bc05e80-68c3-11eb-9e31-94aa1f9c59b2.png"></td>
    <td align="center"><img src="https://user-images.githubusercontent.com/16078332/107119521-0d8a2200-68c3-11eb-96e0-12ca2a0cebb5.png"></td>
  </tr>
</table>

## License

GNU General Public License, version 3 (GPLv3)
