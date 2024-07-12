# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

import contextlib
import math
import os

from nvitop.api import NA, colored, host, set_color  # noqa: F401 # pylint: disable=unused-import
from nvitop.gui.library.widestring import WideString


USERNAME = 'N/A'
with contextlib.suppress(ImportError, OSError):
    USERNAME = host.getuser()

SUPERUSER = False
with contextlib.suppress(AttributeError, OSError):
    if host.WINDOWS:
        import ctypes

        SUPERUSER = bool(ctypes.windll.shell32.IsUserAnAdmin())
    else:
        try:
            SUPERUSER = os.geteuid() == 0
        except AttributeError:
            SUPERUSER = os.getuid() == 0

HOSTNAME = host.hostname()
if host.WSL:
    HOSTNAME = f'{HOSTNAME} (WSL)'

USERCONTEXT = f'{USERNAME}@{HOSTNAME}'


LARGE_INTEGER = 65536


def cut_string(s, maxlen, padstr='...', align='left'):
    assert align in {'left', 'right'}

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
def make_bar(prefix, percent, width, *, extra_text=''):
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
            text = f'{min(round(percent), 100):d}%'.replace('100%', 'MAX')
    else:
        bar += '░' * (width - len(bar) - 4)
        text = 'N/A'
    if extra_text and len(f'{bar} {text} {extra_text}') <= width:
        return f'{bar} {text}'.ljust(width - len(extra_text) - 1) + f' {extra_text}'
    return f'{bar} {text}'.ljust(width)
