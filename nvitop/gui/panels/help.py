# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

from ...version import __version__
from ..lib import Displayable


HELP = '''nvitop {} - (C) Xuehai Pan, 2021.
Released under the GNU GPLv3 License.

GPU Process Type: C: Compute, G: Graphics, X: Mixed.

 Arrows: scroll process list                a f c: change display mode
    Esc: clear process selection             ^C I: interrupt selected process
   Home: scroll process list to left most       K: kill selected process
    End: scroll process list to right most      T: terminate selected process
      e: show process environment               h: show this help screen
   F5 r: force refresh window                   q: quit

  on oN: sort by GPU-ID                     os oS: sort by %SM
  op oP: sort by PID                        oc oC: sort by %CPU
  ou oU: sort by USER                       om oM: sort by %MEM
  og oG: sort by GPU-MEM                    ot oT: sort by TIME
    , .: select sort column                     /: invert sort order

Press any key to return.
'''.format(__version__)


class HelpPanel(Displayable):
    def __init__(self, win, root):
        super().__init__(win, root)

        self.infos = HELP.strip().splitlines()

        self.width = max(map(len, self.infos))
        self.height = len(self.infos)

    def draw(self):
        if not self.need_redraw:
            return

        self.color_reset()

        for y, line in enumerate(self.infos, start=self.y):
            self.addstr(y, self.x, line)

        self.color_at(self.y, self.x, width=self.width, fg='cyan', attr='bold')
        self.color_at(self.y + 1, self.x, width=self.width, fg='cyan', attr='bold')

        self.color_at(self.y + self.height - 1, self.x, width=self.width, fg='cyan', attr='bold')

        self.color_at(self.y + 3, self.x, width=17, fg='white', attr='bold')
        for dx in (18, 30, 43):
            self.color_at(self.y + 3, self.x + dx, width=1, fg='magenta', attr='bold')
        for dx in (21, 33, 48):
            self.color_at(self.y + 3, self.x + dx, width=1, attr='underline')

        for dy in range(5, 11):
            self.color_at(self.y + dy, self.x + 1, width=7, fg='cyan', attr='bold')
            self.color_at(self.y + dy, self.x + 44, width=6, fg='cyan', attr='bold')
        for dy in range(12, 16):
            self.color_at(self.y + dy, self.x + 1, width=7, fg='blue', attr='bold')
            self.color_at(self.y + dy, self.x + 44, width=6, fg='blue', attr='bold')
        self.color_at(self.y + 16, self.x + 1, width=7, fg='magenta', attr='bold')
        self.color_at(self.y + 16, self.x + 44, width=6, fg='magenta', attr='bold')
        for dy in (6, 7, 8):
            self.color_at(self.y + dy, self.x + 44, width=6, fg='red', attr='bold')

    def finalize(self):
        self.need_redraw = False

    def press(self, key):
        self.root.keymaps.use_keymap('help')
        self.root.press(key)
