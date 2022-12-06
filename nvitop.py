#!/usr/bin/env python3
#
# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""An interactive NVIDIA-GPU process viewer and beyond, the one-stop solution for GPU process management."""

import sys

from nvitop.cli import main  # pylint: disable=no-name-in-module


if __name__ == '__main__':
    sys.exit(main())
