# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# This file is originally part of ranger, the console file manager. https://github.com/ranger/ranger
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring

from unicodedata import east_asian_width


ASCIIONLY = set(map(chr, range(1, 128)))
NARROW = 1
WIDE = 2
WIDE_SYMBOLS = set('WF')


def utf_char_width(string):
    """Return the width of a single character"""

    if east_asian_width(string) in WIDE_SYMBOLS:
        return WIDE
    return NARROW


def string_to_charlist(string):
    """Return a list of characters with extra empty strings after wide chars"""

    if ASCIIONLY.issuperset(string):
        return list(string)
    result = []
    for char in string:
        result.append(char)
        if east_asian_width(char) in WIDE_SYMBOLS:
            result.append('')
    return result


def wcslen(string):
    """Return the length of a string with wide chars"""

    return len(WideString(string))


class WideString:  # pylint: disable=too-few-public-methods
    def __init__(self, string='', chars=None):
        if isinstance(string, WideString):
            string = string.string

        try:
            self.string = str(string)
        except UnicodeEncodeError:
            self.string = string.encode('latin-1', 'ignore')
        if chars is None:
            self.chars = string_to_charlist(string)
        else:
            self.chars = chars

    def __add__(self, other):
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

    def __radd__(self, other):
        """
        >>> ('bc' + WideString('afd')).chars
        ['b', 'c', 'a', 'f', 'd']
        """

        if isinstance(other, str):
            return WideString(other + self.string)
        if isinstance(other, WideString):
            return WideString(other.string + self.string, other.chars + self.chars)
        return NotImplemented

    def __iadd__(self, other):
        new = self + other
        self.string = new.string
        self.chars = new.chars
        return self

    def __str__(self):
        return self.string

    def __repr__(self):
        return '<{} {!r}>'.format(self.__class__.__name__, self.string)

    def __eq__(self, other):
        if not isinstance(other, (str, WideString)):
            raise TypeError
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.string)

    def __getslice__(self, start, stop):
        """
        >>> WideString('asdf')[1:3]
        <WideString 'sd'>
        >>> WideString('asdf')[1:-100]
        <WideString ''>
        >>> WideString('モヒカン')[2:4]
        <WideString 'ヒ'>
        >>> WideString('モヒカン')[2:5]
        <WideString 'ヒ '>
        >>> WideString('モabカン')[2:5]
        <WideString 'ab '>
        >>> WideString('モヒカン')[1:5]
        <WideString ' ヒ '>
        >>> WideString('モヒカン')[:]
        <WideString 'モヒカン'>
        >>> WideString('aモ')[0:3]
        <WideString 'aモ'>
        >>> WideString('aモ')[0:2]
        <WideString 'a '>
        >>> WideString('aモ')[0:1]
        <WideString 'a'>
        """

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

    def __getitem__(self, item):
        """
        >>> WideString('asdf')[2]
        <WideString 'd'>
        >>> WideString('……')[0]
        <WideString '…'>
        >>> WideString('……')[1]
        <WideString '…'>
        """

        if isinstance(item, slice):
            assert item.step is None or item.step == 1
            return self.__getslice__(item.start, item.stop)
        return self.__getslice__(item, item + 1)

    def __len__(self):
        """
        >>> len(WideString('poo'))
        3
        >>> len(WideString('モヒカン'))
        8
        """
        return len(self.chars)

    def ljust(self, width, fillchar=' '):
        """
        >>> WideString('poo').ljust(2)
        <WideString 'poo'>
        >>> WideString('poo').ljust(5)
        <WideString 'poo  '>
        >>> WideString('モヒカン').ljust(10)
        <WideString 'モヒカン  '>
        """

        if width > len(self):
            return WideString(self.string + fillchar * width)[:width]
        return self

    def rjust(self, width, fillchar=' '):
        """
        >>> WideString('poo').rjust(2)
        <WideString 'poo'>
        >>> WideString('poo').rjust(5)
        <WideString '  poo'>
        >>> WideString('モヒカン').rljust(10)
        <WideString '  モヒカン'>
        """

        if width > len(self):
            return WideString(fillchar * width + self.string)[-width:]
        return self

    def center(self, width, fillchar=' '):
        """
        >>> WideString('poo').center(2)
        <WideString 'poo'>
        >>> WideString('poo').center(5)
        <WideString ' poo '>
        >>> WideString('モヒカン').center(10)
        <WideString ' モヒカン '>
        """

        if width > len(self):
            left_width = (width - len(self)) // 2
            right_width = width - left_width
            return WideString(fillchar * left_width + self.string + fillchar * right_width)[:width]
        return self

    def strip(self, chars=None):
        """
        >>> WideString('  poo  ').strip()
        <WideString 'poo'>
        >>> WideString('  モヒカン  ').strip()
        <WideString 'モヒカン'>
        """

        return WideString(self.string.strip(chars))

    def lstrip(self, chars=None):
        """
        >>> WideString('  poo  ').lstrip()
        <WideString 'poo  '>
        >>> WideString('  モヒカン  ').lstrip()
        <WideString 'モヒカン  '>
        """

        return WideString(self.string.lstrip(chars))

    def rstrip(self, chars=None):
        """
        >>> WideString('  poo  ').rstrip()
        <WideString '  poo'>
        >>> WideString('  モヒカン  ').rstrip()
        <WideString '  モヒカン'>
        """

        return WideString(self.string.rstrip(chars))
