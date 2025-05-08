# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

from __future__ import annotations

import contextlib
import math
import os
from typing import Literal

from nvitop.api import (
    NA,
    GiB,
    NaType,
    Snapshot,
    bytes2human,
    colored,
    set_color,
    timedelta2human,
    ttl_cache,
)
from nvitop.api.host import WINDOWS as IS_WINDOWS
from nvitop.api.host import WINDOWS_SUBSYSTEM_FOR_LINUX
from nvitop.tui.library.host import getuser, hostname
from nvitop.tui.library.widestring import WideString


__all__ = [
    'HOSTNAME',
    'IS_SUPERUSER',
    'IS_WINDOWS',
    'IS_WINDOWS_SUBSYSTEM_FOR_LINUX',
    'IS_WSL',
    'LARGE_INTEGER',
    'NA',
    'USERNAME',
    'USER_CONTEXT',
    'GiB',
    'NaType',
    'Snapshot',
    'bytes2human',
    'colored',
    'cut_string',
    'make_bar',
    'set_color',
    'timedelta2human',
    'ttl_cache',
]


USERNAME: str = getuser()

IS_SUPERUSER: bool = False
with contextlib.suppress(AttributeError, OSError):
    if IS_WINDOWS:
        import ctypes

        IS_SUPERUSER = bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    else:
        try:
            IS_SUPERUSER = os.geteuid() == 0
        except AttributeError:
            IS_SUPERUSER = os.getuid() == 0

HOSTNAME: str = hostname()
IS_WINDOWS_SUBSYSTEM_FOR_LINUX = IS_WSL = bool(WINDOWS_SUBSYSTEM_FOR_LINUX)
if IS_WSL:
    HOSTNAME = f'{HOSTNAME} (WSL)'

USER_CONTEXT: str = f'{USERNAME}@{HOSTNAME}'


LARGE_INTEGER: int = 65536


def cut_string(
    s: object,
    maxlen: int,
    padstr: str = '...',
    align: Literal['left', 'right'] = 'left',
) -> str:
    assert align in ('left', 'right')

    if not isinstance(s, str):
        s = str(s)
    s = WideString(s)
    padstr = WideString(padstr)

    if len(s) <= maxlen:
        return str(s)
    if len(padstr) >= maxlen:
        return str(padstr[:maxlen])
    if align == 'left':
        return str(s[: maxlen - len(padstr)] + padstr)
    return str(padstr + s[-(maxlen - len(padstr)) :])


# pylint: disable=disallowed-name
def make_bar(prefix: str, percent: float | str, width: int, *, extra_text: str = '') -> str:
    bar = f'{prefix}: '
    if percent != NA and not (isinstance(percent, float) and not math.isfinite(percent)):
        if isinstance(percent, str) and percent.endswith('%'):
            percent = percent.replace('%', '')
            percent = float(percent) if '.' in percent else int(percent)
        percentage = max(0.0, min(float(percent) / 100.0, 1.0))
        quotient, remainder = divmod(max(1, round(8 * (width - len(bar) - 4) * percentage)), 8)
        bar += '█' * quotient
        if remainder > 0:
            bar += ' ▏▎▍▌▋▊▉'[remainder]
        if isinstance(percent, float) and len(f'{bar} {percent:.1f}%') <= width:
            text = f'{percent:.1f}%'
        else:
            text = f'{min(round(percent), 100):d}%'.replace('100%', 'MAX')  # type: ignore[arg-type]
    else:
        bar += '░' * (width - len(bar) - 4)
        text = 'N/A'
    if extra_text and len(f'{bar} {text} {extra_text}') <= width:
        return f'{bar} {text}'.ljust(width - len(extra_text) - 1) + f' {extra_text}'
    return f'{bar} {text}'.ljust(width)
