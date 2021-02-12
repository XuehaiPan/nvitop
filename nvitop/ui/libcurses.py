# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import contextlib
import curses
import signal


@contextlib.contextmanager
def libcurses():
    win = curses.initscr()
    win.nodelay(True)
    win.leaveok(True)
    win.keypad(True)

    curses.noecho()
    curses.cbreak()
    curses.curs_set(False)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    curses.mouseinterval(0)
    curses.ungetmouse(0, 0, 0, 0, 0)

    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass

    # Push a Ctrl+C (ascii value 3) to the curses getch stack
    def interrupt_handler(signalnum, frame): curses.ungetch(3)  # pylint: disable=multiple-statements,unused-argument

    # Simulate a ^C press in curses when an interrupt is caught
    signal.signal(signal.SIGINT, interrupt_handler)

    try:
        yield win
    finally:
        curses.endwin()
