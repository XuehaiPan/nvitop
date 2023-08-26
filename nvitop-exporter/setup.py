#!/usr/bin/env python3

"""Setup script for ``nvitop-exporter``."""

import pathlib
import re
import sys

from setuptools import setup


HERE = pathlib.Path(__file__).absolute().parent
VERSION_FILE = HERE / 'nvitop_exporter' / 'version.py'

sys.path.insert(0, str(VERSION_FILE.parent))
# pylint: disable-next=import-error,wrong-import-position
import version  # noqa


VERSION_CONTENT = None

try:
    if not version.__release__:
        try:
            VERSION_CONTENT = VERSION_FILE.read_text(encoding='utf-8')
            VERSION_FILE.write_text(
                data=re.sub(
                    r"""__version__\s*=\s*('[^']+'|"[^"]+")""",
                    f'__version__ = {version.__version__!r}',
                    string=VERSION_CONTENT,
                ),
                encoding='utf-8',
            )
        except OSError:
            VERSION_CONTENT = None

    setup(
        name='nvitop-exporter',
        version=version.__version__,
    )
finally:
    if VERSION_CONTENT is not None:
        with VERSION_FILE.open(mode='wt', encoding='utf-8', newline='') as file:
            file.write(VERSION_CONTENT)
