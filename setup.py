#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import pathlib
import sys

from setuptools import setup


VERSION_FILE = pathlib.Path(__file__).absolute().parent / 'nvitop' / 'version.py'

try:
    from nvitop import version
except ImportError:
    sys.path.insert(0, str(VERSION_FILE.parent))
    import version


VERSION_CONTENT = None
if not version.__release__:
    import re

    VERSION_CONTENT = VERSION_FILE.read_text(encoding='UTF-8')
    VERSION_FILE.write_text(data=re.sub(r"""__version__\s*=\s*('[^']+'|"[^"]+")""",
                                        r"__version__ = '{}'".format(version.__version__),
                                        string=VERSION_CONTENT),
                            encoding='UTF-8')


setup(
    name='nvitop',
    version=version.__version__,
    description=version.__doc__,
    author=version.__author__,
    author_email=version.__email__,
)


if VERSION_CONTENT is not None:
    VERSION_FILE.write_text(data=VERSION_CONTENT, encoding='UTF-8')
