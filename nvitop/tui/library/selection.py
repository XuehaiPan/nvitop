# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import signal
import time
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

from nvitop.api import NA, Snapshot
from nvitop.tui.library import host
from nvitop.tui.library.utils import IS_SUPERUSER, IS_WINDOWS, LARGE_INTEGER, USERNAME


if TYPE_CHECKING:
    from collections.abc import Callable

    from nvitop.tui.library.displayable import Displayable
    from nvitop.tui.library.process import GpuProcess, HostProcess


__all__ = ['Selection']


class Selection:  # pylint: disable=too-many-instance-attributes
    def __init__(self, displayable: Displayable) -> None:
        self.tagged: WeakValueDictionary[int, GpuProcess | HostProcess] = WeakValueDictionary()
        self.displayable: Displayable = displayable
        self.index: int | None = None
        self.within_window: bool = True
        self._process: GpuProcess | HostProcess | None = None
        self._username: str | None = None
        self._ident: tuple | None = None

    @property
    def identity(self) -> tuple:
        if self._ident is None:
            self._ident = self.process._ident  # type: ignore[union-attr] # pylint: disable=protected-access
        return self._ident

    @property
    def process(self) -> GpuProcess | HostProcess | None:
        return self._process

    @process.setter
    def process(self, process: Snapshot | GpuProcess | HostProcess | None) -> None:
        if isinstance(process, Snapshot):
            process = process.real
        self._process = process  # type: ignore[assignment]
        self._ident = None

    @property
    def pid(self) -> int | None:
        try:
            return self.identity[0]
        except TypeError:
            return None

    @property
    def username(self) -> str | None:
        if self._username is None:
            try:
                self._username = self.process.username()  # type: ignore[union-attr]
            except host.PsutilError:
                self._username = NA
        return self._username

    def move(self, direction: int = 0) -> None:
        if direction == 0:
            return

        processes = self.displayable.snapshots  # type: ignore[attr-defined]
        old_index = self.index

        if len(processes) > 0:
            if not self.is_set():
                if abs(direction) < LARGE_INTEGER:
                    self.index = 0 if direction > 0 else len(processes) - 1
                else:
                    self.index = len(processes) - 1 if direction > 0 else 0
            else:
                self.index = min(max(0, self.index + direction), len(processes) - 1)  # type: ignore[operator]
            self.process = processes[self.index]

            if old_index is not None:
                direction -= self.index - old_index
            else:
                direction = 0

            if direction != 0 and self.displayable.NAME == 'process':  # type: ignore[attr-defined]
                self.displayable.parent.move(direction)  # type: ignore[union-attr]
        else:
            self.clear()

    def owned(self) -> bool:
        if not self.is_set():
            return False
        return IS_SUPERUSER or self.username == USERNAME

    def tag(self) -> None:
        if self.is_set():
            try:
                del self.tagged[self.pid]  # type: ignore[arg-type]
            except KeyError:
                self.tagged[self.pid] = self.process  # type: ignore[index,assignment]

    def processes(self) -> tuple[GpuProcess | HostProcess, ...]:
        if len(self.tagged) > 0:
            return tuple(sorted(self.tagged.values(), key=lambda p: p.pid))
        if self.owned() and self.within_window:
            return (self.process,)  # type: ignore[return-value]
        return ()

    def foreach(self, func: Callable[[GpuProcess | HostProcess], None]) -> None:
        flag = False
        for process in self.processes():
            try:
                func(process)
            except host.PsutilError:  # noqa: PERF203
                pass
            else:
                flag = True

        if flag:
            time.sleep(0.25)
        self.clear()

    def send_signal(self, sig: int) -> None:
        self.foreach(lambda process: process.send_signal(sig))

    def interrupt(self) -> None:
        try:
            # pylint: disable-next=no-member
            self.send_signal(signal.SIGINT if not IS_WINDOWS else signal.CTRL_C_EVENT)  # type: ignore[attr-defined]
        except SystemError:
            pass

    def terminate(self) -> None:
        self.foreach(lambda process: process.terminate())

    def kill(self) -> None:
        self.foreach(lambda process: process.kill())

    def reset(self) -> None:
        self.index = None
        self.within_window = True
        self._process = None
        self._username = None
        self._ident = None

    def clear(self) -> None:
        self.tagged.clear()
        self.reset()

    def is_set(self) -> bool:
        return self.process is not None

    def is_same(self, process: Snapshot | GpuProcess | HostProcess) -> bool:
        if isinstance(process, Snapshot):
            process = process.real

        try:
            return self.identity == process._ident  # pylint: disable=protected-access
        except (AttributeError, TypeError):
            pass

        return False

    def is_same_on_host(self, process: Snapshot | GpuProcess | HostProcess) -> bool:
        if isinstance(process, Snapshot):
            process = process.real

        try:
            return self.identity[:2] == process._ident[:2]  # pylint: disable=protected-access
        except (AttributeError, TypeError):
            pass

        return False

    def is_tagged(self, process: Snapshot | GpuProcess | HostProcess) -> bool:
        return process.pid in self.tagged
