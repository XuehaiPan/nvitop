# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring

from typing import TYPE_CHECKING as _TYPE_CHECKING

from nvitop.tui.library.curses import ascii  # pylint: disable=redefined-builtin


try:
    from curses import *  # noqa: F403 # pylint: disable=redefined-builtin
except ImportError:
    from nvitop.tui.library.curses._curses import *  # noqa: F403 # pylint: disable=redefined-builtin

    # Copied from the CPython repository.
    # https://github.com/python/cpython/blob/HEAD/Lib/curses/__init__.py

    # Some constants, most notably the ACS_* ones, are only added to the C
    # _curses module's dictionary after initscr() is called.  (Some
    # versions of SGI's curses don't define values for those constants
    # until initscr() has been called.)  This wrapper function calls the
    # underlying C initscr(), and then copies the constants from the
    # _curses module to the curses package's dictionary.  Don't do 'from
    # curses import *' if you'll be needing the ACS_* constants.

    def initscr():
        import os
        import sys

        from . import _curses  # noqa: TID252

        # we call setupterm() here because it raises an error
        # instead of calling exit() in error cases.
        setupterm(term=os.getenv('TERM', 'unknown'), fd=sys.__stdout__.fileno())
        stdscr = _curses.initscr()
        globals().update(
            {
                key: value
                for key, value in vars(_curses).items()
                if key.startswith('ACS_') or key in ('LINES', 'COLS')
            },
        )

        return stdscr

    # This is a similar wrapper for start_color(), which adds the COLORS and
    # COLOR_PAIRS variables which are only available after start_color() is
    # called.

    def start_color():
        from . import _curses  # noqa: TID252

        retval = _curses.start_color()
        if hasattr(_curses, 'COLORS'):
            globals()['COLORS'] = _curses.COLORS
        if hasattr(_curses, 'COLOR_PAIRS'):
            globals()['COLOR_PAIRS'] = _curses.COLOR_PAIRS
        return retval

    # Wrapper for the entire curses-based application.  Runs a function which
    # should be the rest of your curses-based application.  If the application
    # raises an exception, wrapper() will restore the terminal to a sane state so
    # you can read the resulting traceback.

    def wrapper(func, /, *args, **kwds):
        """Wrapper function that initializes curses and calls another function,
        restoring normal keyboard/screen behavior on error.

        The callable object 'func' is then passed the main window 'stdscr'
        as its first argument, followed by any other arguments passed to
        wrapper().
        """

        try:
            # Initialize curses
            stdscr = initscr()

            # Turn off echoing of keys, and enter cbreak mode,
            # where no buffering is performed on keyboard input
            noecho()
            cbreak()

            # In keypad mode, escape sequences for special keys
            # (like the cursor keys) will be interpreted and
            # a special value like curses.KEY_LEFT will be returned
            stdscr.keypad(True)

            # Start color, too.  Harmless if the terminal doesn't have
            # color; user can test with has_color() later on.  The try/catch
            # works around a minor bit of over-conscientiousness in the curses
            # module -- the error return from C start_color() is ignorable.
            try:
                start_color()
            except:  # noqa: B001,E722,RUF100
                pass

            return func(stdscr, *args, **kwds)
        finally:
            # Set everything back to normal
            if 'stdscr' in locals():
                stdscr.keypad(False)
                echo()
                nocbreak()
                endwin()


if _TYPE_CHECKING:
    try:
        from curses import window as CursesWindow  # noqa: N812
    except ImportError:
        from nvitop.tui.library.curses._curses import CursesWindow
