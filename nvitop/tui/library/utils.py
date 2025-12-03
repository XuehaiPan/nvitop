# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring,invalid-name

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
    'make_bar_chart',
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


# pylint: disable-next=too-many-arguments
def make_bar_chart(
    prefix: str,
    percent: float | str,
    width: int,
    *,
    extra_text: str = '',
    swap_text: bool = False,
    extra_blank: str = '',
) -> str:
    bar_chart = f'{prefix}: '
    if percent != NA and not (isinstance(percent, float) and not math.isfinite(percent)):
        if isinstance(percent, str) and percent.endswith('%'):
            percent = percent.replace('%', '')
            percent = float(percent) if '.' in percent else int(percent)
        percentage = max(0.0, min(float(percent) / 100.0, 1.0))
        quotient, remainder = divmod(
            max(1, round(8 * (width - len(bar_chart) - 4) * percentage)),
            8,
        )
        bar_chart += '█' * quotient
        if remainder > 0:
            bar_chart += ' ▏▎▍▌▋▊▉'[remainder]
        if isinstance(percent, float) and len(f'{bar_chart} {percent:.1f}%') <= width:
            text = f'{percent:.1f}%'
        else:
            text = f'{min(round(percent), 100):d}%'.replace('100%', 'MAX')  # type: ignore[arg-type]
    else:
        text = 'N/A'
        if (
            extra_text
            and 'N/A' not in extra_text.upper()
            and swap_text
            and len(bar_chart) + len(extra_text) + 2 <= width
        ):
            text, extra_text = extra_text, ''
        bar_chart += '░' * (width - len(bar_chart) - len(text) - 1)
    if extra_text:
        if len(f'{bar_chart} {text} {extra_blank}{extra_text}') <= width:
            if swap_text:
                text, extra_text = extra_text, text
            return (
                f'{bar_chart} {text}'.ljust(width - len(extra_blank) - len(extra_text) - 1)
                + f' {extra_blank}{extra_text}'
            )
        if len(f'{bar_chart} {extra_text}') <= width and swap_text:
            return f'{bar_chart} {extra_text}'.ljust(width)
    return f'{bar_chart} {text}'.ljust(width)
