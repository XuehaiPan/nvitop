# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring

import getpass
import math
import os
import platform

from nvitop.core import NA, colored, host, set_color  # pylint: disable=unused-import
from nvitop.gui.library.widestring import WideString


LARGE_INTEGER = 65536


def cut_string(s, maxlen, padstr='...', align='left'):
    assert align in ('left', 'right')

    if not isinstance(s, str):
        s = str(s)
    s = WideString(s)

    if len(s) <= maxlen:
        return str(s)
    if align == 'left':
        return str(s[: maxlen - len(padstr)] + padstr)
    return str(padstr + s[-(maxlen - len(padstr)) :])


# pylint: disable=disallowed-name
def make_bar(prefix, percent, width):
    bar = '{}: '.format(prefix)
    if percent != NA and not (isinstance(percent, float) and not math.isfinite(percent)):
        if isinstance(percent, str) and percent.endswith('%'):
            percent = percent.replace('%', '')
            percent = float(percent) if '.' in percent else int(percent)
        percentage = max(0.0, min(float(percent) / 100.0, 1.0))
        quotient, remainder = divmod(max(1, round(8 * (width - len(bar) - 4) * percentage)), 8)
        bar += '█' * quotient
        if remainder > 0:
            bar += ' ▏▎▍▌▋▊▉'[remainder]
        if isinstance(percent, float) and len('{} {:.1f}%'.format(bar, percent)) <= width:
            bar += ' {:.1f}%'.format(percent)
        else:
            bar += ' {:d}%'.format(min(round(percent), 100)).replace('100%', 'MAX')
    else:
        bar += '░' * (width - len(bar) - 4) + ' N/A'
    return bar.ljust(width)


try:
    USERNAME = getpass.getuser()
except ModuleNotFoundError:
    USERNAME = os.getlogin()

if host.WINDOWS:
    import ctypes

    SUPERUSER = bool(ctypes.windll.shell32.IsUserAnAdmin())
else:
    try:
        SUPERUSER = os.geteuid() == 0
    except AttributeError:
        try:
            SUPERUSER = os.getuid() == 0
        except AttributeError:
            SUPERUSER = False

HOSTNAME = platform.node()
if host.WSL:
    HOSTNAME = '{} (WSL)'.format(HOSTNAME)

USERCONTEXT = '{}@{}'.format(USERNAME, HOSTNAME)
