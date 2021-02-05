# nvhtop

An interactive Nvidia-GPU process viewer. (Under progress)

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
pip install -e git+https://github.com/XuehaiPan/nvhtop.git#egg=nvhtop
```

Or, download and pip install:

```bash
git clone --depth=1 https://github.com/XuehaiPan/nvhtop.git
cd nvhtop
pip3 install .
```

## Usage

Query the device status. The output is similar to `nvidia-smi`, but has been enriched and colorized.

```bash
nvhtop
```

Run as a resource monitor, like `htop`.

```bash
# Automatically configure the display mode according to the terminal size
nvhtop.py --monitor
nvhtop.py --monitor auto

# Forcibly display as `full` mode
nvhtop.py --monitor full

# Forcibly display as `compact` mode
nvhtop.py --monitor compact
```

Press `q` to return to the terminal.

Type `nvhtop --help` for more information:

```
usage: nvhtop [-h] [-m [{auto,full,compact}]]

A interactive Nvidia-GPU process viewer.

optional arguments:
  -h, --help        show this help message and exit
  -m [{auto,full,compact}], --monitor [{auto,full,compact}]
                    Run as a resource monitor. Continuously report query data,
                    rather than the default of just once.
                    If no argument is specified, the default mode `auto` is used.
```

## Screenshots

Example output of `nvhtop`:

<img width="600" alt="Screenshot" src="https://user-images.githubusercontent.com/16078332/106898060-af2c3a80-672e-11eb-9ab6-1ccbaa6292b5.png">

Example output of `nvhtop --monitor`:

<table>
  <tr>
    <td align="center">Full</td>
    <td align="center">Compact</td>
  </tr>
  <tr valign="top">
    <td align="center">
      <img src="https://user-images.githubusercontent.com/16078332/106898073-b18e9480-672e-11eb-812f-85951a8d98cc.png">
    </td>
    <td align="center">
      <img src="https://user-images.githubusercontent.com/16078332/106898074-b2272b00-672e-11eb-9351-3301240b2d42.png">
    </td>
  </tr>
</table>
