# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

from nvitop.core import Device
from nvitop.version import __version__
from nvitop.gui.library import Displayable, MouseEvent


HELP_TEMPLATE = '''nvitop {} - (C) Xuehai Pan, 2021.
Released under the GNU GPLv3 License.

GPU Process Type: C: Compute, G: Graphics, X: Mixed.

Device coloring rules by loading intensity:
  - GPU utilization: light < {:2d}% <= moderate < {:2d}% <= heavy.
  - MEM utilization: light < {:2d}% <= moderate < {:2d}% <= heavy.

 Arrows: scroll process list               Ctrl-C I: interrupt selected process
    Esc: clear process selection                  K: kill selected process
 Ctrl-A: scroll process list to left most         T: terminate selected process
 Ctrl-E: scroll process list to right most
   Home: select the first process             a f c: change display mode
    End: select the last process                h ?: show this help screen
      e: show process environment              F5 r: force refresh window
      t: toggle tree-view screen                  q: quit

  Wheel: scroll process list            Shift-Wheel: scroll horizontally
    Tab: scroll process list             Ctrl-Wheel: fast scroll ({}x)

  on oN: sort by GPU-INDEX                    os oS: sort by %SM
  op oP: sort by PID                          oc oC: sort by %CPU
  ou oU: sort by USER                         om oM: sort by %MEM
  og oG: sort by GPU-MEM                      ot oT: sort by TIME
    , .: select sort column                       /: invert sort order

Press any key to return.
'''


class HelpScreen(Displayable):
    NAME = 'help'

    def __init__(self, win, root):
        super().__init__(win, root)

        self.previous_screen = 'main'

        HELP = HELP_TEMPLATE.format(__version__,
                                    *Device.GPU_UTILIZATION_THRESHOLDS,
                                    *Device.MEMORY_UTILIZATION_THRESHOLDS,
                                    MouseEvent.CTRL_SCROLLWHEEL_MULTIPLIER)

        self.infos = HELP.strip().splitlines()

        self.x, self.y = root.x, root.y
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

        self.color_at(self.y + 5, self.x, width=21, fg='white', attr='bold')
        for dy in (6, 7):
            self.color_at(self.y + dy, self.x + 21, width=5, fg='green', attr='bold | italic')
            self.color_at(self.y + dy, self.x + 36, width=8, fg='yellow', attr='bold | italic')
            self.color_at(self.y + dy, self.x + 54, width=5, fg='red', attr='bold | italic')

        for dy in range(9, 17):
            self.color_at(self.y + dy, self.x, width=8, fg='cyan', attr='bold')
            self.color_at(self.y + dy, self.x + 44, width=8, fg='green', attr='bold')
        for dy in (18, 19):
            self.color_at(self.y + dy, self.x, width=8, fg='cyan', attr='bold')
            self.color_at(self.y + dy, self.x + 40, width=12, fg='cyan', attr='bold')
        for dy in range(21, 25):
            self.color_at(self.y + dy, self.x, width=8, fg='blue', attr='bold')
            self.color_at(self.y + dy, self.x + 44, width=8, fg='blue', attr='bold')
        self.color_at(self.y + 25, self.x, width=8, fg='magenta', attr='bold')
        self.color_at(self.y + 25, self.x + 44, width=8, fg='magenta', attr='bold')
        for dy in (9, 10, 11):
            self.color_at(self.y + dy, self.x + 43, width=9, fg='red', attr='bold')

    def press(self, key):
        self.root.keymaps.use_keymap('help')
        self.root.press(key)
