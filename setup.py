#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import os
import sys

from setuptools import setup

try:
    from nvitop import version
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nvitop'))
    import version


setup(
    name='nvitop',
    version=version.__version__,
    description=version.__doc__,
    author=version.__author__,
    author_email=version.__email__,
)
