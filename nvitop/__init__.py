# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""An interactive NVIDIA-GPU process viewer, the one-stop solution for GPU process management."""

from . import core
from .core import *
from .version import __version__


__all__ = core.__all__.copy()
