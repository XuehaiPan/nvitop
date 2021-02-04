# nvhtop

An interactive Nvidia-GPU process viewer. (Under progress)

## Installation

```bash
# Clone the repo
git clone --depth=1 https://github.com/XuehaiPan/nvhtop.git
cd nvhtop

# Install dependencies
pip3 install -r requirements.txt
```

## Usage

Query the device status. The output is similar to `nvidia-smi`, but has been enriched and colorized.

```bash
./nvhtop.py
```

Run as a resource monitor, like `htop`.

```bash
# Automatically configure the display mode according to the terminal size
./nvhtop.py --monitor
./nvhtop.py --monitor auto

# Forcibly display as `full` mode
./nvhtop.py --monitor full

# Forcibly display as `compact` mode
./nvhtop.py --monitor compact
```

Press `q` to return to the terminal.

Type `./nvhtop.py --help` for more information:

```
usage: nvhtop.py [-h] [-m [{auto,full,compact}]]

A interactive Nvidia-GPU process viewer.

optional arguments:
  -h, --help        show this help message and exit
  -m [{auto,full,compact}], --monitor [{auto,full,compact}]
                    Run as a resource monitor. Continuously report query data,
                    rather than the default of just once.
                    If no argument is specified, the default mode `auto` is used.
```

For convenience, you can create a symbolic link in `~/.local`.

```bash
mkdir -p ~/.local/bin
ln -sf "$PWD/nvhtop.py" ~/.local/bin/nvhtop
```

## Screenshots

Example output of `nvhtop.py`:

<img width="600" alt="Screenshot" src="https://user-images.githubusercontent.com/16078332/106898060-af2c3a80-672e-11eb-9ab6-1ccbaa6292b5.png">

Example output of `nvhtop.py --monitor`:

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
