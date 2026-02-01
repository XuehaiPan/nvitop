# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-function-docstring,import-outside-toplevel,invalid-name

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING


HAS_CURSES_MODULE: bool = True
try:
    import curses
except ImportError:
    HAS_CURSES_MODULE = False
else:
    del curses

    from curses import *  # noqa: F403 # pylint: disable=redefined-builtin
    from curses import ascii  # pylint: disable=redefined-builtin


if not HAS_CURSES_MODULE:
    # pylint: disable-next=redefined-builtin
    from nvitop.tui.library.curses import ascii  # type: ignore[no-redef]
    from nvitop.tui.library.curses._curses import *  # type: ignore[assignment] # noqa: F403

    if _TYPE_CHECKING:
        from collections.abc import Callable as _Callable
        from typing import TypeVar as _TypeVar  # pylint: disable=ungrouped-imports
        from typing_extensions import Concatenate as _Concatenate  # Python 3.10+
        from typing_extensions import ParamSpec as _ParamSpec  # Python 3.10+

        # pylint: disable-next=ungrouped-imports
        from nvitop.tui.library.curses._curses import (  # type: ignore[assignment]
            CursesWindow as window,  # noqa: N813
        )

        _P = _ParamSpec('_P')
        _T = _TypeVar('_T')

    # Copied from the CPython repository.
    # https://github.com/python/cpython/blob/HEAD/Lib/curses/__init__.py

    # Some constants, most notably the ACS_* ones, are only added to the C
    # _curses module's dictionary after initscr() is called.  (Some
    # versions of SGI's curses don't define values for those constants
    # until initscr() has been called.)  This wrapper function calls the
    # underlying C initscr(), and then copies the constants from the
    # _curses module to the curses package's dictionary.  Don't do 'from
    # curses import *' if you'll be needing the ACS_* constants.
    def initscr() -> window:  # pylint: disable=function-redefined
        import os
        import sys

        from nvitop.tui.library.curses import _curses

        assert sys.__stdout__ is not None

        # we call setupterm() here because it raises an error
        # instead of calling exit() in error cases.
        _curses.setupterm(term=os.getenv('TERM', 'unknown'), fd=sys.__stdout__.fileno())
        stdscr = _curses.initscr()
        globals().update(
            {
                key: value
                for key, value in vars(_curses).items()
                if key.startswith('ACS_') or key in ('LINES', 'COLS')
            },
        )

        return stdscr  # type: ignore[return-value]

    # This is a similar wrapper for start_color(), which adds the COLORS and
    # COLOR_PAIRS variables which are only available after start_color() is
    # called.
    def start_color() -> None:  # pylint: disable=function-redefined
        from nvitop.tui.library.curses import _curses

        retval = _curses.start_color()  # type: ignore[func-returns-value] # pylint: disable=assignment-from-no-return
        if hasattr(_curses, 'COLORS'):
            globals()['COLORS'] = _curses.COLORS
        if hasattr(_curses, 'COLOR_PAIRS'):
            globals()['COLOR_PAIRS'] = _curses.COLOR_PAIRS
        return retval

    # Wrapper for the entire curses-based application.  Runs a function which
    # should be the rest of your curses-based application.  If the application
    # raises an exception, wrapper() will restore the terminal to a sane state so
    # you can read the resulting traceback.
    def wrapper(  # pylint: disable=function-redefined
        func: _Callable[_Concatenate[window, _P], _T],
        /,
        *args: _P.args,
        **kwds: _P.kwargs,
    ) -> _T:
        """Wrapper function that initializes curses and calls another function,
        restoring normal keyboard/screen behavior on error.

        The callable object 'func' is then passed the main window 'stdscr'
        as its first argument, followed by any other arguments passed to
        wrapper().
        """
        from nvitop.tui.library.curses import _curses

        try:
            # Initialize curses
            stdscr = initscr()

            # Turn off echoing of keys, and enter cbreak mode,
            # where no buffering is performed on keyboard input
            _curses.noecho()
            _curses.cbreak()

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
            except:  # noqa: E722,S110,RUF100 # pylint: disable=bare-except
                pass

            return func(stdscr, *args, **kwds)
        finally:
            # Set everything back to normal
            if 'stdscr' in locals():
                stdscr.keypad(False)
                _curses.echo()
                _curses.nocbreak()
                _curses.endwin()

else:
    if _TYPE_CHECKING:
        from curses import window  # pylint: disable=ungrouped-imports
