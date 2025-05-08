# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=wrong-spelling-in-comment
# Vendored from the `termcolor` package: https://github.com/termcolor/termcolor
# ==============================================================================
# Copyright (c) 2008-2011 Volvox Development Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: Konstantin Lepa <konstantin.lepa@gmail.com>
# ==============================================================================
"""ANSI color formatting for output in terminal."""

from __future__ import annotations

import io
import os
import sys
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from collections.abc import Iterable

    Attribute = Literal[
        'bold',
        'dark',
        'underline',
        'blink',
        'reverse',
        'concealed',
        'strike',
    ]
    Highlight = Literal[
        'on_black',
        'on_grey',
        'on_red',
        'on_green',
        'on_yellow',
        'on_blue',
        'on_magenta',
        'on_cyan',
        'on_light_grey',
        'on_dark_grey',
        'on_light_red',
        'on_light_green',
        'on_light_yellow',
        'on_light_blue',
        'on_light_magenta',
        'on_light_cyan',
        'on_white',
    ]
    Color = Literal[
        'black',
        'grey',
        'red',
        'green',
        'yellow',
        'blue',
        'magenta',
        'cyan',
        'light_grey',
        'dark_grey',
        'light_red',
        'light_green',
        'light_yellow',
        'light_blue',
        'light_magenta',
        'light_cyan',
        'white',
    ]


__all__ = ['colored', 'cprint']


if os.name == 'nt':  # Windows
    try:
        from colorama import init
    except ImportError:
        pass
    else:
        init()


ATTRIBUTES: dict[Attribute, int] = {
    'bold': 1,
    'dark': 2,
    'underline': 4,
    'blink': 5,
    'reverse': 7,
    'concealed': 8,
    'strike': 9,
}

HIGHLIGHTS: dict[Highlight, int] = {
    'on_black': 40,
    'on_grey': 40,  # Actually black but kept for backwards compatibility
    'on_red': 41,
    'on_green': 42,
    'on_yellow': 43,
    'on_blue': 44,
    'on_magenta': 45,
    'on_cyan': 46,
    'on_light_grey': 47,
    'on_dark_grey': 100,
    'on_light_red': 101,
    'on_light_green': 102,
    'on_light_yellow': 103,
    'on_light_blue': 104,
    'on_light_magenta': 105,
    'on_light_cyan': 106,
    'on_white': 107,
}

COLORS: dict[Color, int] = {
    'black': 30,
    'grey': 30,  # Actually black but kept for backwards compatibility
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'magenta': 35,
    'cyan': 36,
    'light_grey': 37,
    'dark_grey': 90,
    'light_red': 91,
    'light_green': 92,
    'light_yellow': 93,
    'light_blue': 94,
    'light_magenta': 95,
    'light_cyan': 96,
    'white': 97,
}


RESET = '\033[0m'


# pylint: disable-next=too-many-return-statements
def _can_do_color(
    *,
    no_color: bool | None = None,
    force_color: bool | None = None,
) -> bool:
    """Check env vars and for tty/dumb terminal."""
    # First check overrides:
    # "User-level configuration files and per-instance command-line arguments should
    # override $NO_COLOR. A user should be able to export $NO_COLOR in their shell
    # configuration file as a default, but configure a specific program in its
    # configuration file to specifically enable color."
    # https://no-color.org
    if no_color is not None and no_color:
        return False
    if force_color is not None and force_color:
        return True

    # Then check env vars:
    if 'ANSI_COLORS_DISABLED' in os.environ:
        return False
    if 'NO_COLOR' in os.environ:
        return False
    if 'FORCE_COLOR' in os.environ:
        return True

    # Then check system:
    if os.environ.get('TERM') == 'dumb':
        return False
    if not hasattr(sys.stdout, 'fileno'):
        return False

    try:
        return os.isatty(sys.stdout.fileno())
    except io.UnsupportedOperation:
        return sys.stdout.isatty()


# pylint: disable-next=too-many-arguments
def colored(
    text: Any,
    /,
    color: Color | None = None,
    on_color: Highlight | None = None,
    attrs: Iterable[Attribute] | None = None,
    *,
    no_color: bool | None = None,
    force_color: bool | None = None,
) -> str:
    """Colorize text.

    Available text colors:
        black, red, green, yellow, blue, magenta, cyan, white,
        light_grey, dark_grey, light_red, light_green, light_yellow, light_blue,
        light_magenta, light_cyan.

    Available text highlights:
        on_black, on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white,
        on_light_grey, on_dark_grey, on_light_red, on_light_green, on_light_yellow,
        on_light_blue, on_light_magenta, on_light_cyan.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Example:
        colored('Hello, World!', 'red', 'on_black', ['bold', 'blink'])
        colored('Hello, World!', 'green')
    """
    result = str(text)
    if not _can_do_color(no_color=no_color, force_color=force_color):
        return result

    fmt_str = '\033[%dm%s'
    if color is not None:
        result = fmt_str % (COLORS[color], result)

    if on_color is not None:
        result = fmt_str % (HIGHLIGHTS[on_color], result)

    if attrs is not None:
        for attr in attrs:
            result = fmt_str % (ATTRIBUTES[attr], result)

    result += RESET

    return result


# pylint: disable-next=too-many-arguments
def cprint(
    text: object,
    /,
    color: Color | None = None,
    on_color: Highlight | None = None,
    attrs: Iterable[Attribute] | None = None,
    *,
    no_color: bool | None = None,
    force_color: bool | None = None,
    **kwargs: Any,
) -> None:
    """Print colorized text.

    It accepts arguments of print function.
    """
    print(
        colored(
            text,
            color,
            on_color,
            attrs,
            no_color=no_color,
            force_color=force_color,
        ),
        **kwargs,
    )
