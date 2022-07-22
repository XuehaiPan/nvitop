#!/usr/bin/env python3

# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""CUDA visible devices selection tool.

Usage:

.. code-block:: bash

    nvisel --min-count 2 --max-count 3 --min-free-memory 5GiB --max-gpu-utilization 60

    export CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="$(nvisel -c 1 -f 10GiB)"

    export CUDA_VISIBLE_DEVICES="$(nvisel -O uuid -c 2 -f 10GiB)"

    nvisel -0 -O uuid -c 2 -f 4GiB | xargs -0 -I {} nvidia-smi --id={} --query-gpu=index,memory.free --format=csv
"""

import sys

from nvitop.select import main  # pylint: disable=no-name-in-module


if __name__ == '__main__':
    sys.exit(main())
