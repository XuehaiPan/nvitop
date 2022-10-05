# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import signal
import time
from weakref import WeakValueDictionary

from nvitop.core import NA, Snapshot, host
from nvitop.gui.library.utils import LARGE_INTEGER, SUPERUSER, USERNAME


class Selection:  # pylint: disable=too-many-instance-attributes
    def __init__(self, panel):
        self.tagged = WeakValueDictionary()
        self.panel = panel
        self.index = None
        self.within_window = True
        self._process = None
        self._username = None
        self._ident = None

    @property
    def identity(self):
        if self._ident is None:
            self._ident = self.process._ident  # pylint: disable=protected-access
        return self._ident

    @property
    def process(self):
        return self._process

    @process.setter
    def process(self, process):
        if isinstance(process, Snapshot):
            process = process.real
        self._process = process
        self._ident = None

    @property
    def pid(self):
        try:
            return self.identity[0]
        except TypeError:
            return None

    @property
    def username(self):
        if self._username is None:
            try:
                self._username = self.process.username()
            except host.PsutilError:
                self._username = NA
        return self._username

    def move(self, direction=0):
        if direction == 0:
            return

        processes = self.panel.snapshots
        old_index = self.index

        if len(processes) > 0:
            if not self.is_set():
                if abs(direction) < LARGE_INTEGER:
                    self.index = 0 if direction > 0 else len(processes) - 1
                else:
                    self.index = len(processes) - 1 if direction > 0 else 0
            else:
                self.index = min(max(0, self.index + direction), len(processes) - 1)
            self.process = processes[self.index]

            if old_index is not None:
                direction -= self.index - old_index
            else:
                direction = 0

            if direction != 0 and self.panel.NAME == 'process':
                self.panel.parent.move(direction)
        else:
            self.clear()

    def owned(self):
        if not self.is_set():
            return False
        if SUPERUSER:
            return True
        return self.username == USERNAME

    def tag(self):
        if self.is_set():
            try:
                del self.tagged[self.pid]
            except KeyError:
                self.tagged[self.pid] = self.process

    def processes(self):
        if len(self.tagged) > 0:
            return tuple(sorted(self.tagged.values(), key=lambda p: p.pid))
        if self.owned() and self.within_window:
            return (self.process,)
        return ()

    def foreach(self, func):
        flag = False
        for process in self.processes():
            try:
                func(process)
            except host.PsutilError:
                pass
            else:
                flag = True

        if flag:
            time.sleep(0.25)
        self.clear()

    def send_signal(self, sig):
        self.foreach(lambda process: process.send_signal(sig))

    def interrupt(self):
        try:
            self.send_signal(
                signal.SIGINT
                if not host.WINDOWS
                else signal.CTRL_C_EVENT  # pylint: disable=no-member
            )
        except SystemError:
            pass

    def terminate(self):
        self.foreach(lambda process: process.terminate())

    def kill(self):
        self.foreach(lambda process: process.kill())

    def reset(self):
        self.index = None
        self.within_window = True
        self._process = None
        self._username = None
        self._ident = None

    def clear(self):
        self.tagged.clear()
        self.reset()

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

    def is_tagged(self, process):
        return process.pid in self.tagged
