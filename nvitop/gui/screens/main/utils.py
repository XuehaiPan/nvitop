# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import signal
import time
from collections import namedtuple

from nvitop.core import host, Snapshot
from nvitop.gui.library import CURRENT_USER, IS_SUPERUSER


class Selected(object):
    def __init__(self, panel):
        self.panel = panel
        self.index = None
        self.within_window = True
        self._proc = None
        self._ident = None

    @property
    def identity(self):
        if self._ident is None:
            self._ident = self.process._ident  # pylint: disable=protected-access
        return self._ident

    @property
    def process(self):
        return self._proc

    @process.setter
    def process(self, process):
        if isinstance(process, Snapshot):
            process = process.real
        self._proc = process
        self._ident = None

    @property
    def pid(self):
        try:
            return self.identity[0]
        except TypeError:
            return None

    def move(self, direction=0):
        if direction == 0:
            return

        processes = self.panel.snapshots
        if len(processes) > 0:
            if not self.is_set():
                if abs(direction) < 1024:
                    self.index = (0 if direction > 0 else len(processes) - 1)
                else:
                    self.index = (len(processes) - 1 if direction > 0 else 0)
            else:
                self.index = min(max(0, self.index + direction), len(processes) - 1)
            self.process = processes[self.index]
        else:
            self.clear()

    def owned(self):
        if not self.is_set():
            return False
        if IS_SUPERUSER:
            return True
        try:
            return self.process.username() == CURRENT_USER
        except host.PsutilError:
            return False

    def send_signal(self, sig):
        if self.owned() and self.within_window:
            try:
                self.process.send_signal(sig)
            except host.PsutilError:
                pass
            else:
                time.sleep(0.5)
                if not self.process.is_running():
                    self.clear()

    def interrupt(self):
        return self.send_signal(signal.SIGINT)

    def terminate(self):
        if self.owned() and self.within_window:
            try:
                self.process.terminate()
            except host.PsutilError:
                pass
            else:
                time.sleep(0.5)
                self.clear()

    def kill(self):
        if self.owned() and self.within_window:
            try:
                self.process.kill()
            except host.PsutilError:
                pass
            else:
                time.sleep(0.5)
                self.clear()

    def clear(self):
        self.__init__(self.panel)

    reset = clear

    def is_set(self):
        return self.process is not None

    __bool__ = is_set

    def is_same(self, process):
        try:
            return self.identity == process._ident  # pylint: disable=protected-access
        except (AttributeError, TypeError):
            pass

        return False

    __eq__ = is_same

    def is_same_on_host(self, process):
        try:
            return self.identity[:2] == process._ident[:2]  # pylint: disable=protected-access
        except (AttributeError, TypeError):
            pass

        return False


Order = namedtuple('Order', ['key', 'reverse', 'offset', 'column', 'previous', 'next'])
