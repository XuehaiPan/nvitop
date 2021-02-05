# nvhtop

An interactive Nvidia-GPU process viewer.

## Requirements

- Python 3.5+
- curses
- nvidia-ml-py
- psutil
- cachetools
- termcolor

## Installation

From GitHub:

```bash
$ pip install git+https://github.com/XuehaiPan/nvhtop.git#egg=nvhtop
```

Or, download and `pip install`:

```bash
$ git clone --depth=1 https://github.com/XuehaiPan/nvhtop.git
$ cd nvhtop
$ pip install .
```

## Usage

Query the device status. The output is similar to `nvidia-smi`, but has been enriched and colorized.

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
usage: nvhtop [-h] [-m [{auto,full,compact}]] [--mem-util-thresh th th]
              [--gpu-util-thresh th th]

A interactive Nvidia-GPU process viewer.

optional arguments:
  -h, --help            show this help message and exit
  -m [{auto,full,compact}], --monitor [{auto,full,compact}]
                        Run as a resource monitor. Continuously report query
                        data, rather than the default of just once. If no
                        argument is specified, the default mode `auto` is
                        used.
  --mem-util-thresh th th
                        Thresholds of GPU memory utilization to distinguish
                        load intensity, (1 <= th <= 99, defaults: 10 80)
  --gpu-util-thresh th th
                        Thresholds of GPU utilization to distinguish load
                        intensity, (1 <= th <= 99, defaults: 10 75)
```

## Screenshots

Example output of `nvhtop`:

<img width="600" alt="Screenshot" src="https://user-images.githubusercontent.com/16078332/107054137-b32d8a80-680a-11eb-9a0e-dd9975fd9ecc.png">

Example output of `nvhtop --monitor`:

<table>
  <tr>
    <td align="center">Full</td>
    <td align="center">Compact</td>
  </tr>
  <tr valign="top">
    <td align="center">
      <img src="https://user-images.githubusercontent.com/16078332/107054291-e6701980-680a-11eb-8da0-8d59dfce0ed7.png">
    </td>
    <td align="center">
      <img src="https://user-images.githubusercontent.com/16078332/107054190-c3de0080-680a-11eb-8016-4fb958d4bbc4.png">
    </td>
  </tr>
</table>
