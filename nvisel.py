#!/usr/bin/env python3

# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""CUDA visible devices selection tool.

Usage:

.. code-block:: bash

    # All devices but sorted
    nvisel       # or use `python3 -m nvitop.select`

    # A simple example to select 4 devices
    nvisel -n 4  # or use `python3 -m nvitop.select -n 4`

    # Select available devices that satisfy the given constraints
    nvisel --min-count 2 --max-count 3 --min-free-memory 5GiB --max-gpu-utilization 60

    # Set `CUDA_VISIBLE_DEVICES` environment variable using `nvisel`
    export CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="$(nvisel -c 1 -f 10GiB)"

    # Use UUID strings in `CUDA_VISIBLE_DEVICES` environment variable
    export CUDA_VISIBLE_DEVICES="$(nvisel -O uuid -c 2 -f 5000M)"

    # Pipe output to other shell utilities
    nvisel -0 -O uuid -c 2 -f 4GiB | xargs -0 -I {} nvidia-smi --id={} --query-gpu=index,memory.free --format=csv
"""

import sys

from nvitop.select import main  # pylint: disable=no-name-in-module


if __name__ == '__main__':
    sys.exit(main())
