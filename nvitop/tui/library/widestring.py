# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# This file is originally part of ranger, the console file manager. https://github.com/ranger/ranger
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from unicodedata import east_asian_width


if TYPE_CHECKING:
    from typing_extensions import Self  # Python 3.11+


__all__ = ['WideString', 'wcslen']


ASCIIONLY: frozenset[str] = frozenset(map(chr, range(1, 128)))
NARROW: Literal[1] = 1
WIDE: Literal[2] = 2
WIDE_SYMBOLS: frozenset[str] = frozenset('WF')


def utf_char_width(string: str) -> Literal[1, 2]:
    """Return the width of a single character."""
    if east_asian_width(string) in WIDE_SYMBOLS:
        return WIDE
    return NARROW


def string_to_charlist(string: str) -> list[str]:
    """Return a list of characters with extra empty strings after wide chars."""
    if ASCIIONLY.issuperset(string):
        return list(string)
    result = []
    for char in string:
        result.append(char)
        if east_asian_width(char) in WIDE_SYMBOLS:
            result.append('')
    return result


def wcslen(string: str | WideString) -> int:
    # pylint: disable=wrong-spelling-in-docstring
    """Return the length of a string with wide chars.

    >>> wcslen('poo')
    3
    >>> wcslen('十百千万')
    8
    >>> wcslen('a十')
    3
    """
    return len(WideString(string))


class WideString:  # pylint: disable=wrong-spelling-in-docstring
    def __init__(self, string: str | WideString = '', chars: list[str] | None = None) -> None:
        self.string: str = str(string)
        self.chars: list[str] = string_to_charlist(self.string) if chars is None else chars

    def __add__(self, other: object) -> WideString:
        """
        >>> (WideString('a') + WideString('b')).string
        'ab'
        >>> (WideString('a') + WideString('b')).chars
        ['a', 'b']
        >>> (WideString('afd') + 'bc').chars
        ['a', 'f', 'd', 'b', 'c']
        """
        if isinstance(other, str):
            return WideString(self.string + other)
        if isinstance(other, WideString):
            return WideString(self.string + other.string, self.chars + other.chars)
        return NotImplemented

    def __radd__(self, other: object) -> WideString:
        """
        >>> ('bc' + WideString('afd')).chars
        ['b', 'c', 'a', 'f', 'd']
        """
        if isinstance(other, str):
            return WideString(other + self.string)
        if isinstance(other, WideString):
            return WideString(other.string + self.string, other.chars + self.chars)
        return NotImplemented

    def __iadd__(self, other: object) -> Self:
        new = self + other
        self.string = new.string
        self.chars = new.chars
        return self

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.string!r}>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (str, WideString)):
            raise TypeError
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(self.string)

    def __getitem__(self, item: int | slice) -> WideString:
        """
        >>> WideString('asdf')[2]
        <WideString 'd'>
        >>> WideString('……')[0]
        <WideString '…'>
        >>> WideString('……')[1]
        <WideString '…'>
        >>> WideString('asdf')[1:3]
        <WideString 'sd'>
        >>> WideString('asdf')[1:-100]
        <WideString ''>
        >>> WideString('十百千万')[2:4]
        <WideString '百'>
        >>> WideString('十百千万')[2:5]
        <WideString '百 '>
        >>> WideString('十ab千万')[2:5]
        <WideString 'ab '>
        >>> WideString('十百千万')[1:5]
        <WideString ' 百 '>
        >>> WideString('十百千万')[:]
        <WideString '十百千万'>
        >>> WideString('a十')[0:3]
        <WideString 'a十'>
        >>> WideString('a十')[0:2]
        <WideString 'a '>
        >>> WideString('a十')[0:1]
        <WideString 'a'>
        """
        if isinstance(item, slice):
            assert item.step is None or item.step == 1
            start, stop = item.start, item.stop
        else:
            assert isinstance(item, int)
            start, stop = item, item + 1

        length = len(self)

        if stop is None or stop > length:
            stop = length
        if stop < 0:
            stop = max(0, length + stop)
        if start is None:
            start = 0
        if start < 0:
            start = max(0, length + start)
        if start >= length or start >= stop:
            return WideString('')
        if stop < length and self.chars[stop] == '':
            if self.chars[start] == '':
                return WideString(' ' + ''.join(self.chars[start : stop - 1]) + ' ')
            return WideString(''.join(self.chars[start : stop - 1]) + ' ')
        if self.chars[start] == '':
            return WideString(' ' + ''.join(self.chars[start : stop - 1]))
        return WideString(''.join(self.chars[start:stop]))

    def __len__(self) -> int:
        """
        >>> len(WideString('poo'))
        3
        >>> len(WideString('十百千万'))
        8
        """
        return len(self.chars)

    def ljust(self, width: int, fillchar: str = ' ') -> WideString:
        """
        >>> WideString('poo').ljust(2)
        <WideString 'poo'>
        >>> WideString('poo').ljust(5)
        <WideString 'poo  '>
        >>> WideString('十百千万').ljust(10)
        <WideString '十百千万  '>
        """
        if width > len(self):
            return WideString(self.string + fillchar * width)[:width]
        return self

    def rjust(self, width: int, fillchar: str = ' ') -> WideString:
        """
        >>> WideString('poo').rjust(2)
        <WideString 'poo'>
        >>> WideString('poo').rjust(5)
        <WideString '  poo'>
        >>> WideString('十百千万').rjust(10)
        <WideString '  十百千万'>
        """
        if width > len(self):
            return WideString(fillchar * width + self.string)[-width:]
        return self

    def center(self, width: int, fillchar: str = ' ') -> WideString:
        """
        >>> WideString('poo').center(2)
        <WideString 'poo'>
        >>> WideString('poo').center(5)
        <WideString ' poo '>
        >>> WideString('十百千万').center(10)
        <WideString ' 十百千万 '>
        """
        if width > len(self):
            left_width = (width - len(self)) // 2
            right_width = width - left_width
            return WideString(fillchar * left_width + self.string + fillchar * right_width)[:width]
        return self

    def strip(self, chars: str | None = None) -> WideString:
        """
        >>> WideString('  poo  ').strip()
        <WideString 'poo'>
        >>> WideString('  十百千万  ').strip()
        <WideString '十百千万'>
        """
        return WideString(self.string.strip(chars))

    def lstrip(self, chars: str | None = None) -> WideString:
        """
        >>> WideString('  poo  ').lstrip()
        <WideString 'poo  '>
        >>> WideString('  十百千万  ').lstrip()
        <WideString '十百千万  '>
        """
        return WideString(self.string.lstrip(chars))

    def rstrip(self, chars: str | None = None) -> WideString:
        """
        >>> WideString('  poo  ').rstrip()
        <WideString '  poo'>
        >>> WideString('  十百千万  ').rstrip()
        <WideString '  十百千万'>
        """
        return WideString(self.string.rstrip(chars))


if __name__ == '__main__':
    import doctest

    doctest.testmod()
