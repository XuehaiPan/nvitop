# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from nvitop.tui.library import Displayable, Selection


if TYPE_CHECKING:
    from nvitop.tui.screens.main import MainScreen
    from nvitop.tui.tui import TUI


__all__ = ['BasePanel', 'BaseSelectablePanel']


class BasePanel(Displayable):
    """Base class for all panels."""

    root: TUI
    parent: MainScreen

    NAME: ClassVar[str]
    SNAPSHOT_INTERVAL: ClassVar[float] = 0.5


class BaseSelectablePanel(BasePanel):
    """Base class for all selectable panels."""

    selection: Selection
