# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from nvitop.gui.library import Device, Displayable, MouseEvent
from nvitop.version import __version__


HELP_TEMPLATE = '''nvitop {} - (C) Xuehai Pan, 2021-2022.
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

      on oN: sort by GPU-INDEX                os oS: sort by %SM
      op oP: sort by PID                      oc oC: sort by %CPU
      ou oU: sort by USER                     om oM: sort by %MEM
      og oG: sort by GPU-MEM                  ot oT: sort by TIME
        , .: select sort column                   /: invert sort order

Press any key to return.
'''


class HelpScreen(Displayable):  # pylint: disable=too-many-instance-attributes
    NAME = 'help'

    def __init__(self, win, root):
        super().__init__(win, root)

        self.infos = (
            HELP_TEMPLATE.format(
                __version__,
                *Device.GPU_UTILIZATION_THRESHOLDS,
                *Device.MEMORY_UTILIZATION_THRESHOLDS,
                MouseEvent.CTRL_SCROLLWHEEL_MULTIPLIER,
            )
            .strip()
            .splitlines()
        )
        self.color_matrix = {
            9: ('green', 'green'),
            10: ('green', 'green'),
            12: ('cyan', 'yellow'),
            13: ('cyan', 'yellow'),
            14: ('cyan', 'red'),
            15: (None, 'red'),
            16: ('cyan', 'red'),
            **{dy: ('cyan', 'green') for dy in range(17, 20)},
            **{dy: ('blue', 'blue') for dy in range(21, 23)},
            **{dy: ('blue', 'blue') for dy in range(24, 28)},
            28: ('magenta', 'magenta'),
        }

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
            if right is not None:
                self.color_at(self.y + dy, self.x + 39, width=13, fg=right, attr='bold')

    def press(self, key):
        self.root.keymaps.use_keymap('help')
        self.root.press(key)
