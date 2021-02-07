# This file is part of nvhtop, the interactive Nvidia-GPU process viewer.
# License: GNU GPL version 3.

import contextlib
import curses
import signal


@contextlib.contextmanager
def libcurses():
    win = curses.initscr()
    win.nodelay(True)
    curses.noecho()
    curses.cbreak()
    curses.curs_set(False)

    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass

    def interrupt_handler(signum, frame):
        # Push a Ctrl+C (ascii value 3) to the curses getch stack
        curses.ungetch(3)
    # Simulate a ^C press in curses when an interrupt is caught
    signal.signal(signal.SIGINT, interrupt_handler)

    try:
        yield win
    finally:
        curses.endwin()
