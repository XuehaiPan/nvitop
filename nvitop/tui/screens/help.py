# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any, ClassVar

from nvitop.tui.library import LAYOUT_WIDTH, Device, MouseEvent
from nvitop.tui.screens.base import BaseScreen
from nvitop.version import __version__


if TYPE_CHECKING:
    import curses

    from nvitop.tui.tui import TUI


__all__ = ['HelpScreen']


HELP_TEMPLATE = r"""nvitop {} - (C) Xuehai Pan, 2021-2026.
Released under the GNU GPLv3 License.

GPU Process Type: C: Compute, G: Graphics, X: Mixed.

Device coloring rules by loading intensity:
  - GPU utilization: light < {:2d}% <= moderate < {:2d}% <= heavy.
  - GPU-MEM percent: light < {:2d}% <= moderate < {:2d}% <= heavy.

      a f c: change display mode                h ?: show this help screen
       F5 r: force refresh window                 q: quit

     Arrows: scroll process list              Space: tag/untag current process
       Home: select the first process           Esc: clear process selection
        End: select the last process       Ctrl-C I: interrupt selected process
                                                  K: kill selected process
   Ctrl-A ^: scroll to left most                  T: terminate selected process
   Ctrl-E $: scroll to right most                 e: show process environment
   PageUp [: scroll entire screen up              t: toggle tree-view screen
 PageDown ]: scroll entire screen down        Enter: show process metrics

      Wheel: scroll process list        Shift-Wheel: scroll horizontally
        Tab: scroll process list         Ctrl-Wheel: fast scroll ({}x)

      on oN: sort by GPU-INDEX
      op oP: sort by PID                      ob oB: sort by %GMBW
      ou oU: sort by USER                     oc oC: sort by %CPU
      og oG: sort by GPU-MEM                  om oM: sort by %MEM
      os oS: sort by %SM                      ot oT: sort by TIME
        , .: select sort column                   /: invert sort order

Press any key to return.
"""

# The narrow layout lists the keybindings of HELP_TEMPLATE in a single column
HELP_KEYBINDING_ROWS = range(9, 30)
HELP_RIGHT_COLUMN = 39
HELP_KEY_WIDTH = 11
NO_BREAK_SPACE = '\xa0'
# Highlights in the header rows: (row, text to find, offset in the text, width, attributes)
HELP_HEADER_MARKS: tuple[tuple[int, str, int, int, dict[str, Any]], ...] = (
    (3, 'GPU Process Type:', 0, 17, {'attr': 'bold'}),
    *(
        (3, f'{letter}: {name}', offset, 1, attrs)
        for letter, name, underline in (
            ('C', 'Compute', 3),
            ('G', 'Graphics', 3),
            ('X', 'Mixed', 5),
        )
        for offset, attrs in (
            (0, {'fg': 'magenta', 'attr': 'bold'}),
            (underline, {'attr': 'underline'}),
        )
    ),
    (5, 'Device coloring rules', 0, 21, {'attr': 'bold'}),
    *(
        (row, word, 0, len(word), {'fg': color, 'attr': 'bold | italic'})
        for row in (6, 7)
        for word, color in (('light', 'green'), ('moderate', 'yellow'), ('heavy', 'red'))
    ),
)


class HelpScreen(BaseScreen):  # pylint: disable=too-many-instance-attributes
    NAME: ClassVar[str] = 'help'

    def __init__(self, *, win: curses.window, root: TUI) -> None:
        super().__init__(win, root)

        self.infos: list[str] = (
            HELP_TEMPLATE.format(
                __version__,
                *Device.GPU_UTILIZATION_THRESHOLDS,
                *Device.MEMORY_UTILIZATION_THRESHOLDS,
                MouseEvent.CTRL_SCROLLWHEEL_MULTIPLIER,
            )
            .strip()
            .splitlines()
        )
        self.color_matrix: dict[int, tuple[str | None, str | None]] = {
            9: ('green', 'green'),
            10: ('green', 'green'),
            12: ('cyan', 'yellow'),
            13: ('cyan', 'yellow'),
            14: ('cyan', 'red'),
            15: (None, 'red'),
            16: ('cyan', 'red'),
            **dict.fromkeys(range(17, 20), ('cyan', 'green')),
            **dict.fromkeys(range(21, 23), ('blue', 'blue')),
            **dict.fromkeys(range(24, 29), ('blue', 'blue')),
            29: ('magenta', 'magenta'),
        }
        # Rows whose right-hand action column is colored red are exactly the
        # kill/terminate/interrupt keybindings — the same set ``--readonly`` disables.
        self.readonly_disabled_rows: frozenset[int] = frozenset(
            row for row, (_, right) in self.color_matrix.items() if right == 'red'
        )

        self.x, self.y = root.x, root.y
        self.width: int = max(map(len, self.infos))
        self.height: int = len(self.infos)

        self.narrow_cache: tuple[int, list[str], list[tuple[int, int, int, dict[str, Any]]]] | None
        self.narrow_cache = None
        self.scroll_offset: int = 0
        self.scrollable: bool = False

    def narrow_content(
        self,
        width: int,
    ) -> tuple[list[str], list[tuple[int, int, int, dict[str, Any]]]]:
        """Return the lines of the narrow layout and their highlights (row, x, width, attributes)."""
        if self.narrow_cache is None or self.narrow_cache[0] != width:
            lines: list[str] = []
            marks: list[tuple[int, int, int, dict[str, Any]]] = []
            self.narrow_header(width, lines, marks)
            self.narrow_keybindings(lines, marks)
            self.narrow_cache = (width, lines, marks)
        return self.narrow_cache[1], self.narrow_cache[2]

    def narrow_header(
        self,
        width: int,
        lines: list[str],
        marks: list[tuple[int, int, int, dict[str, Any]]],
    ) -> None:
        """Wrap the header rows, keeping each "C: Compute" style pair on the same line."""
        for row, info in enumerate(self.infos[: HELP_KEYBINDING_ROWS.start]):
            for letter in 'CGX':
                info = info.replace(f' {letter}: ', f' {letter}:{NO_BREAK_SPACE}')
            indent = ' ' * (len(info) - len(info.lstrip(' -')))
            for line in textwrap.wrap(info, width=width, subsequent_indent=indent) or ['']:
                line = line.replace(NO_BREAK_SPACE, ' ')
                if row in {0, 1}:
                    marks.append((len(lines), 0, len(line), {'fg': 'cyan', 'attr': 'bold'}))
                for mark_row, text, offset, mark_width, attrs in HELP_HEADER_MARKS:
                    start = line.find(text) if mark_row == row else -1
                    if start >= 0:
                        marks.append((len(lines), start + offset, mark_width, attrs))
                lines.append(line)

    def narrow_keybindings(
        self,
        lines: list[str],
        marks: list[tuple[int, int, int, dict[str, Any]]],
    ) -> None:
        """List each group of the two-column table as its left column then its right column."""
        groups: list[list[tuple[int, int, str]]] = [[]]
        for row in HELP_KEYBINDING_ROWS:
            info = self.infos[row]
            if not info.strip():
                groups.append([])
                continue
            for side, entry in enumerate((info[:HELP_RIGHT_COLUMN], info[HELP_RIGHT_COLUMN:])):
                if entry.strip():
                    groups[-1].append((side, row, entry))
        for group in groups:
            for side, row, entry in sorted(group, key=lambda item: item[0]):  # stable sort
                key, _, description = entry.partition(': ')
                attrs = {
                    'fg': self.color_matrix[row][side],
                    'attr': 'bold',
                    'signal': side == 1 and row in self.readonly_disabled_rows,
                }
                marks.append((len(lines), 0, HELP_KEY_WIDTH + 1, attrs))
                lines.append(f'{key.strip():>{HELP_KEY_WIDTH}}: {description.strip()}')
            lines.append('')

    def scroll(self, direction: int) -> bool:
        """Scroll the narrow layout. Return False if there is nothing to scroll."""
        if not self.scrollable:
            return False
        self.scroll_offset += direction
        self.need_redraw = True
        return True

    def draw_narrow(self) -> None:
        assert self.root.termsize is not None
        n_term_lines, n_term_cols = self.root.termsize
        lines, marks = self.narrow_content(n_term_cols - self.x)

        footer = 'Press any key to return.'
        n_rows = len(lines)
        self.scrollable = n_rows + 1 > n_term_lines - self.y
        if self.scrollable:
            footer = 'Up/Down: scroll, any other key: return.'
            n_rows = max(n_term_lines - self.y - 1, 1)
        self.scroll_offset = max(0, min(self.scroll_offset, len(lines) - n_rows))

        for y, line in enumerate(
            lines[self.scroll_offset : self.scroll_offset + n_rows],
            start=self.y,
        ):
            self.addstr(y, self.x, line)
        for row, x, width, attrs in marks:
            y = self.y + row - self.scroll_offset
            if not self.y <= y < self.y + n_rows:
                continue
            attrs = attrs.copy()
            if attrs.pop('signal', False) and self.root.readonly:
                self.color_at(y, self.x + x, width=n_term_cols - x, attr='dim')
            elif attrs.get('fg', '') is not None:
                self.color_at(y, self.x + x, width=width, **attrs)
        self.addstr(self.y + n_rows, self.x, footer)
        self.color_at(self.y + n_rows, self.x, width=len(footer), fg='cyan', attr='bold')

    def draw(self) -> None:
        if not self.need_redraw:
            return

        self.color_reset()

        if self.root.width < LAYOUT_WIDTH:
            self.draw_narrow()
            return
        self.scrollable = False

        for y, line in enumerate(self.infos, start=self.y):
            self.addstr(y, self.x, line)

        self.color_at(self.y, self.x, width=self.width, fg='cyan', attr='bold')
        self.color_at(self.y + 1, self.x, width=self.width, fg='cyan', attr='bold')

        self.color_at(self.y + self.height - 1, self.x, width=self.width, fg='cyan', attr='bold')

        self.color_at(self.y + 3, self.x, width=17, attr='bold')
        for dx in (18, 30, 43):
            self.color_at(self.y + 3, self.x + dx, width=1, fg='magenta', attr='bold')
        for dx in (21, 33, 48):
            self.color_at(self.y + 3, self.x + dx, width=1, attr='underline')

        self.color_at(self.y + 5, self.x, width=21, attr='bold')
        for dy in (6, 7):
            self.color_at(self.y + dy, self.x + 21, width=5, fg='green', attr='bold | italic')
            self.color_at(self.y + dy, self.x + 36, width=8, fg='yellow', attr='bold | italic')
            self.color_at(self.y + dy, self.x + 54, width=5, fg='red', attr='bold | italic')

        for dy, (left, right) in self.color_matrix.items():
            if left is not None:
                self.color_at(self.y + dy, self.x, width=12, fg=left, attr='bold')
            if right is not None and not (self.root.readonly and dy in self.readonly_disabled_rows):
                self.color_at(self.y + dy, self.x + 39, width=13, fg=right, attr='bold')

        if self.root.readonly:
            for dy in self.readonly_disabled_rows:
                self.color_at(self.y + dy, self.x + 39, width=self.width - 39, attr='dim')

    def press(self, key: int) -> bool:
        self.root.keymaps.use_keymap('help')
        return self.root.press(key)
