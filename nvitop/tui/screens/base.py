# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from nvitop.tui.library import DisplayableContainer, Selection


if TYPE_CHECKING:
    from nvitop.tui.tui import TUI


__all__ = ['BaseScreen', 'BaseSelectableScreen']


class BaseScreen(DisplayableContainer):
    """Base class for all screens."""

    root: TUI
    parent: TUI

    NAME: ClassVar[str]


class BaseSelectableScreen(BaseScreen):
    """Base class for all selectable screens."""

    selection: Selection
