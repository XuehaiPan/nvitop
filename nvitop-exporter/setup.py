#!/usr/bin/env python3

"""Setup script for ``nvitop-exporter``."""

from __future__ import annotations

import contextlib
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Generator

from setuptools import setup


if TYPE_CHECKING:
    from types import ModuleType


HERE = Path(__file__).absolute().parent


@contextlib.contextmanager
def vcs_version(name: str, path: Path | str) -> Generator[ModuleType]:
    """Context manager to update version string in a version module."""
    path = Path(path).absolute()
    assert path.is_file()
    module_spec = spec_from_file_location(name=name, location=path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = sys.modules.get(name)
    if module is None:
        module = module_from_spec(module_spec)
        sys.modules[name] = module
    module_spec.loader.exec_module(module)

    if module.__release__:
        yield module
        return

    content = None
    try:
        try:
            content = path.read_text(encoding='utf-8')
            path.write_text(
                data=re.sub(
                    r"""__version__\s*=\s*('[^']+'|"[^"]+")""",
                    f'__version__ = {module.__version__!r}',
                    string=content,
                ),
                encoding='utf-8',
            )
        except OSError:
            content = None

        yield module
    finally:
        if content is not None:
            with path.open(mode='wt', encoding='utf-8', newline='') as file:
                file.write(content)


with vcs_version(
    name='nvitop_exporter.version',
    path=(HERE / 'nvitop_exporter' / 'version.py'),
) as version:
    setup(
        name='nvitop-exporter',
        version=version.__version__,
    )
