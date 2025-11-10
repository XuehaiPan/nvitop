# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# This file is originally part of ranger, the console file manager. https://github.com/ranger/ranger
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import copy
import curses
import curses.ascii
import string
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Dict, Tuple, Union


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing_extensions import TypeAlias  # Python 3.10+


__all__ = [
    'ALT_KEY',
    'ANYKEY',
    'DIGITS',
    'PASSIVE_ACTION',
    'QUANT_KEY',
    'REVERSED_SPECIAL_KEYS',
    'SPECIAL_KEYS',
    'SPECIAL_KEYS',
    'SPECIAL_KEYS_UNCASED',
    'KeyBuffer',
    'KeyMaps',
    'construct_keybinding',
    'normalize_keybinding',
    'parse_keybinding',
]


IntKey: TypeAlias = Union[int, Tuple[int, ...]]


DIGITS: frozenset[int] = frozenset(map(ord, string.digits))

# Arbitrary numbers which are not used with curses.KEY_XYZ
ANYKEY: int = 9001
PASSIVE_ACTION: int = 9002
ALT_KEY: int = 9003
QUANT_KEY: int = 9004


NAMED_SPECIAL_KEYS: OrderedDict[str, int] = OrderedDict(
    [
        ('BS', curses.KEY_BACKSPACE),
        ('Backspace', curses.KEY_BACKSPACE),  # overrides <BS> in REVERSED_SPECIAL_KEYS
        ('Backspace2', curses.ascii.DEL),
        ('Delete', curses.KEY_DC),
        ('S-Delete', curses.KEY_SDC),
        ('Insert', curses.KEY_IC),
        ('CR', ord('\n')),
        ('Return', ord('\n')),
        ('Enter', ord('\n')),  # overrides <CR> and <Return> in REVERSED_SPECIAL_KEYS
        ('Space', ord(' ')),
        ('Escape', curses.ascii.ESC),
        ('Esc', curses.ascii.ESC),  # overrides <Escape> in REVERSED_SPECIAL_KEYS
        ('Down', curses.KEY_DOWN),
        ('Up', curses.KEY_UP),
        ('Left', curses.KEY_LEFT),
        ('Right', curses.KEY_RIGHT),
        ('PageDown', curses.KEY_NPAGE),
        ('PageUp', curses.KEY_PPAGE),
        ('Home', curses.KEY_HOME),
        ('End', curses.KEY_END),
        ('Tab', ord('\t')),
        ('S-Tab', curses.KEY_BTAB),
        ('lt', ord('<')),
        ('gt', ord('>')),
    ],
)
SPECIAL_KEYS: OrderedDict[str, IntKey]
SPECIAL_KEYS_UNCASED: dict[str, IntKey]
REVERSED_SPECIAL_KEYS: dict[IntKey, str]


def _uncase_special_key(key_string: str) -> str:
    """Uncase a special key.

    >>> _uncase_special_key('Esc')
    'esc'

    >>> _uncase_special_key('C-X')
    'c-x'
    >>> _uncase_special_key('C-x')
    'c-x'

    >>> _uncase_special_key('A-X')
    'a-X'
    >>> _uncase_special_key('A-x')
    'a-x'
    """
    uncased = key_string.lower()
    if len(uncased) == 3 and (uncased.startswith(('a-', 'm-'))):
        uncased = f'{uncased[0]}-{key_string[-1]}'
    return uncased


def _special_keys_init() -> None:
    global SPECIAL_KEYS, SPECIAL_KEYS_UNCASED, REVERSED_SPECIAL_KEYS  # pylint: disable=global-statement

    SPECIAL_KEYS = NAMED_SPECIAL_KEYS.copy()  # type: ignore[assignment]
    for key, int_value in NAMED_SPECIAL_KEYS.items():
        SPECIAL_KEYS[f'M-{key}'] = (ALT_KEY, int_value)
        SPECIAL_KEYS[f'A-{key}'] = (ALT_KEY, int_value)  # overrides <M-*> in REVERSED_SPECIAL_KEYS

    for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_!{}[],./':
        SPECIAL_KEYS[f'M-{char}'] = (ALT_KEY, ord(char))
        SPECIAL_KEYS[f'A-{char}'] = (ALT_KEY, ord(char))  # overrides <M-*> in REVERSED_SPECIAL_KEYS

    # We will need to reorder the keys of SPECIAL_KEYS below.
    # For example, <C-j> will override <Enter> in REVERSE_SPECIAL_KEYS,
    # this makes construct_keybinding(parse_keybinding('<CR>')) == '<C-j>'
    for char in 'abcdefghijklmnopqrstuvwxyz_':
        SPECIAL_KEYS[f'C-{char}'] = ord(char) - 96
    SPECIAL_KEYS['C-Space'] = 0

    for n in range(64):
        SPECIAL_KEYS[f'F{n}'] = curses.KEY_F0 + n

    # Very special keys
    SPECIAL_KEYS.update(
        {
            'Alt': ALT_KEY,
            'any': ANYKEY,
            'bg': PASSIVE_ACTION,
            'allow_quantifiers': QUANT_KEY,
        },
    )

    # Reorder the keys of SPECIAL_KEYS
    for key in NAMED_SPECIAL_KEYS:
        SPECIAL_KEYS.move_to_end(key, last=True)

    SPECIAL_KEYS_UNCASED = {_uncase_special_key(k): v for k, v in SPECIAL_KEYS.items()}
    REVERSED_SPECIAL_KEYS = {v: k for k, v in SPECIAL_KEYS.items()}


_special_keys_init()
del _special_keys_init


def parse_keybinding(obj: IntKey | str) -> tuple[int, ...]:  # pylint: disable=too-many-branches
    r"""Translate a keybinding to a sequence of integers
    The letter case of special keys in the keybinding string will be ignored.

    >>> out = parse_keybinding('lol<CR>')
    >>> out
    (108, 111, 108, 10)
    >>> out == (ord('l'), ord('o'), ord('l'), ord('\n'))
    True

    >>> out = parse_keybinding('x<A-Left>')
    >>> out
    (120, 9003, 260)
    >>> out == (ord('x'), ALT_KEY, curses.KEY_LEFT)
    True
    """
    assert isinstance(obj, (tuple, int, str))

    def parse(obj: IntKey | str) -> Generator[int]:  # pylint: disable=too-many-branches
        if isinstance(obj, tuple):
            yield from obj
            return
        if isinstance(obj, int):
            yield obj
            return

        in_brackets = False
        bracket_content: list[str] = []
        for char in obj:
            if in_brackets:
                if char == '>':
                    in_brackets = False
                    key_string = ''.join(bracket_content)
                    try:
                        keys = SPECIAL_KEYS_UNCASED[_uncase_special_key(key_string)]
                        yield from keys  # type: ignore[misc]
                    except KeyError:
                        if key_string.isdigit():
                            yield int(key_string)
                        else:
                            yield ord('<')
                            for bracket_char in bracket_content:
                                yield ord(bracket_char)
                            yield ord('>')
                    except TypeError:
                        yield keys  # type: ignore[misc] # it was not tuple, just an int
                else:
                    bracket_content.append(char)
            elif char == '<':
                in_brackets = True
                bracket_content = []
            else:
                yield ord(char)
        if in_brackets:
            yield ord('<')
            for char in bracket_content:
                yield ord(char)

    return tuple(parse(obj))


def key_to_string(key: IntKey) -> str:
    if key in range(33, 127):
        return chr(key)  # type: ignore[arg-type]
    return f'<{REVERSED_SPECIAL_KEYS.get(key, key)}>'


def construct_keybinding(keys: IntKey) -> str:
    """Do the reverse of parse_keybinding.

    >>> construct_keybinding(parse_keybinding('lol<CR>'))
    'lol<Enter>'

    >>> construct_keybinding(parse_keybinding('x<A-Left>'))
    'x<A-Left>'

    >>> construct_keybinding(parse_keybinding('x<Alt><Left>'))
    'x<A-Left>'
    """
    keys = (keys,) if isinstance(keys, int) else tuple(keys)
    strings = []
    alt_key_on = False
    for key in keys:
        if key == ALT_KEY:
            alt_key_on = True
            continue
        if alt_key_on:
            try:
                strings.append(f'<{REVERSED_SPECIAL_KEYS[ALT_KEY, key]}>')
            except KeyError:
                strings.extend(map(key_to_string, (ALT_KEY, key)))
        else:
            strings.append(key_to_string(key))
        alt_key_on = False

    return ''.join(strings)


def normalize_keybinding(keybinding: str) -> str:
    """Normalize a keybinding to a string.

    >>> normalize_keybinding('lol<CR>')
    'lol<Enter>'

    >>> normalize_keybinding('x<A-Left>')
    'x<A-Left>'

    >>> normalize_keybinding('x<Alt><Left>')
    'x<A-Left>'
    """
    return construct_keybinding(parse_keybinding(keybinding))


KeyMapPointer: TypeAlias = Dict[int, Union['KeyMapPointer', Callable[[], None]]]


class KeyMaps(Dict[str, KeyMapPointer]):
    def __init__(self, keybuffer: KeyBuffer) -> None:
        super().__init__()
        self.keybuffer: KeyBuffer = keybuffer
        self.used_keymap: str | None = None

    def use_keymap(self, keymap_name: str) -> None:
        self.keybuffer.keymap = self.get(keymap_name, {})
        if self.used_keymap != keymap_name:
            self.used_keymap = keymap_name
            self.keybuffer.clear()

    def clear_keymap(self, keymap_name: str) -> KeyMapPointer:
        keymap = self[keymap_name] = {}
        if self.used_keymap == keymap_name:
            self.keybuffer.keymap = keymap
            self.keybuffer.clear()
        return keymap

    def _clean_input(self, context: str, keybinding: str) -> tuple[tuple[int, ...], KeyMapPointer]:
        try:
            pointer = self[context]
        except KeyError:
            self[context] = pointer = {}
        keybinding = keybinding.encode('utf-8').decode('latin-1')
        return parse_keybinding(keybinding), pointer

    def bind(self, context: str, keybinding: str, leaf: Callable[[], None]) -> None:
        keys, pointer = self._clean_input(context, keybinding)
        if not keys:
            return
        last_key = keys[-1]
        for key in keys[:-1]:
            if key in pointer and isinstance(pointer[key], dict):
                pointer = pointer[key]  # type: ignore[assignment]
            else:
                pointer = pointer[key] = {}
        pointer[last_key] = leaf

    def alias(self, context: str, source: str, target: str) -> None:
        clean_source, pointer = self._clean_input(context, source)
        if not source:
            return
        for key in clean_source:
            try:
                pointer = pointer[key]  # type: ignore[assignment]
            except KeyError as ex:  # noqa: PERF203
                raise KeyError(
                    f'Tried to copy the keybinding `{source}`, but it was not found.',
                ) from ex
        try:
            self.bind(context, target, copy.deepcopy(pointer))  # type: ignore[arg-type]
        except TypeError:
            self.bind(context, target, pointer)  # type: ignore[arg-type]

    def unbind(self, context: str, keybinding: str) -> None:
        keys, pointer = self._clean_input(context, keybinding)
        if not keys:
            return

        def unbind_traverse(pointer: KeyMapPointer, keys: list[int], pos: int = 0) -> None:
            if keys[pos] not in pointer:
                return
            if len(keys) > pos + 1 and isinstance(pointer, dict):
                unbind_traverse(pointer[keys[pos]], keys, pos=pos + 1)  # type: ignore[arg-type]
                if not pointer[keys[pos]]:
                    del pointer[keys[pos]]
            elif len(keys) == pos + 1:
                try:
                    del pointer[keys[pos]]
                except KeyError:
                    pass
                try:
                    keys.pop()
                except IndexError:
                    pass

        unbind_traverse(pointer, list(keys))


class KeyBuffer:  # pylint: disable=too-many-instance-attributes
    class QuantifierFinished:  # pylint: disable=too-few-public-methods
        pass

    QUANTIFIER_KEY_FINISHED = QuantifierFinished()

    del QuantifierFinished

    any_key: int = ANYKEY
    passive_key: int = PASSIVE_ACTION
    quantifier_key: int = QUANT_KEY
    excluded_from_anykey: frozenset[int] = frozenset({curses.ascii.ESC})

    def __init__(self, keymap: KeyMapPointer | None = None) -> None:
        self.keymap: KeyMapPointer | None = keymap
        self.keys: list[int] = []
        self.wildcards: list[int] = []
        self.pointer: KeyMapPointer | None = self.keymap
        self.result: Callable[[], None] | None = None
        self.quantifier: int | None = None
        self.finished_parsing_quantifier: bool = False
        self.finished_parsing: bool = False
        self.parse_error: bool = False

        if (
            self.keymap
            and self.quantifier_key in self.keymap
            and self.keymap[self.quantifier_key] is self.QUANTIFIER_KEY_FINISHED  # type: ignore[comparison-overlap]
        ):
            self.finished_parsing_quantifier = True

    def clear(self) -> None:
        self.__init__(self.keymap)  # type: ignore[misc] # pylint: disable=unnecessary-dunder-call

    def add(self, key: int) -> None:
        assert self.pointer is not None
        self.keys.append(key)
        self.result = None
        if not self.finished_parsing_quantifier and key in DIGITS:
            if self.quantifier is None:
                self.quantifier = 0
            self.quantifier = self.quantifier * 10 + key - 48  # (48 = ord('0'))
        else:
            self.finished_parsing_quantifier = True

            moved = True
            if key in self.pointer:
                self.pointer = self.pointer[key]  # type: ignore[assignment]
            elif self.any_key in self.pointer and key not in self.excluded_from_anykey:
                self.wildcards.append(key)
                self.pointer = self.pointer[self.any_key]  # type: ignore[assignment]
            else:
                moved = False

            if moved:
                if isinstance(self.pointer, dict):
                    if self.passive_key in self.pointer:
                        self.result = self.pointer[self.passive_key]  # type: ignore[assignment]
                else:
                    self.result = self.pointer
                    self.finished_parsing = True
            else:
                self.finished_parsing = True
                self.parse_error = True

    def __str__(self) -> str:
        return construct_keybinding(tuple(self.keys))


if __name__ == '__main__':
    import doctest

    doctest.testmod()
