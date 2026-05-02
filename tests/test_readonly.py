# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

"""Regression tests for the ``--readonly`` safety boundary.

The tests cover the two layers that enforce the ``--readonly`` contract
independently of the curses keybinding layer:

1. ``parse_arguments`` accepts ``--readonly`` and honors
   ``NVITOP_MONITOR_MODE=readonly``.
2. ``Selection.send_signal``, ``Selection.terminate``, ``Selection.kill``,
   and ``Selection.interrupt`` are no-ops when their bound ``displayable``'s
   root reports ``readonly=True``, even when invoked directly (e.g., via
   a dialog callback captured before the flag was flipped, or by any
   embedding caller holding a reference to one of those methods).
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

import nvitop.cli
from nvitop.tui.library.selection import Selection


if TYPE_CHECKING:
    from collections.abc import Generator


def _make_selection(*, readonly: bool) -> tuple[Selection, MagicMock]:
    """Build a ``Selection`` bound to a fake ``displayable`` whose root has the given flag.

    Returns the ``Selection`` plus a ``MagicMock`` process so callers can assert
    that no mutating method was invoked when ``readonly`` is ``True``.
    """
    process = MagicMock()
    fake_displayable = SimpleNamespace(root=SimpleNamespace(readonly=readonly))
    selection = Selection(displayable=fake_displayable)  # type: ignore[arg-type]
    # Bypass the snapshots machinery: monkey-patch `processes` so the public
    # methods see exactly one mocked process.
    selection.processes = lambda: (process,)  # type: ignore[method-assign]
    return selection, process


def _parse_with_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    argv: list[str],
    env_value: str | None,
) -> object:
    """Reload ``nvitop.cli`` with the requested env state and parse ``argv``.

    ``nvitop.cli`` captures ``NVITOP_MONITOR_MODE`` into a module-level set at
    import time, so a stale value in the developer's shell would otherwise leak
    into every test. Reloading after each env mutation makes the suite hermetic
    regardless of the launching environment.
    """
    if env_value is None:
        monkeypatch.delenv('NVITOP_MONITOR_MODE', raising=False)
    else:
        monkeypatch.setenv('NVITOP_MONITOR_MODE', env_value)
    importlib.reload(nvitop.cli)
    with patch('sys.argv', argv):
        return nvitop.cli.parse_arguments()


@pytest.fixture(autouse=True, scope='module')
def _restore_cli_module() -> Generator[None, None, None]:
    """Reload ``nvitop.cli`` with a clean env once after the module finishes.

    Tests that mutate ``NVITOP_MONITOR_MODE`` reload the module via
    ``_parse_with_env`` to pick up the new value, and ``monkeypatch`` undoes the
    env mutation at each test's tear-down. The captured module-level set,
    however, lingers until something reloads it. This module-scoped auto-use
    fixture restores ``nvitop.cli`` to match the actual process environment
    after the suite runs, without paying for a reload between every test.
    """
    yield
    importlib.reload(nvitop.cli)


class TestCliReadonly:
    """Surface-level CLI parsing of the ``--readonly`` flag and env var token."""

    def test_flag_sets_readonly_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--readonly`` on ``sys.argv`` flips ``args.readonly`` to ``True``."""
        args = _parse_with_env(monkeypatch, argv=['nvitop', '--readonly'], env_value=None)
        assert args.readonly is True  # type: ignore[attr-defined]

    def test_default_readonly_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No flag and no env var leaves ``args.readonly`` at ``False``."""
        args = _parse_with_env(monkeypatch, argv=['nvitop'], env_value=None)
        assert args.readonly is False  # type: ignore[attr-defined]

    def test_env_var_token_enables_readonly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``NVITOP_MONITOR_MODE=readonly`` enables ``--readonly`` without the CLI flag."""
        args = _parse_with_env(monkeypatch, argv=['nvitop'], env_value='readonly')
        assert args.readonly is True  # type: ignore[attr-defined]

    def test_env_var_comma_separated_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``readonly`` is recognized when combined with other tokens via commas."""
        args = _parse_with_env(monkeypatch, argv=['nvitop'], env_value='readonly,colorful')
        assert args.readonly is True  # type: ignore[attr-defined]
        assert args.colorful is True  # type: ignore[attr-defined]

    def test_unknown_env_token_does_not_enable_readonly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tokens other than ``readonly`` (e.g., ``colorful``) leave the flag off."""
        args = _parse_with_env(monkeypatch, argv=['nvitop'], env_value='colorful')
        assert args.readonly is False  # type: ignore[attr-defined]


class TestSelectionReadonlyBoundary:
    """``Selection`` mutation methods short-circuit when ``root.readonly`` is true."""

    def test_send_signal_blocked(self) -> None:
        """``send_signal`` does not reach ``process.send_signal`` under ``--readonly``."""
        selection, process = _make_selection(readonly=True)
        selection.send_signal(15)
        process.send_signal.assert_not_called()

    def test_terminate_blocked(self) -> None:
        """``terminate`` does not reach ``process.terminate`` under ``--readonly``."""
        selection, process = _make_selection(readonly=True)
        selection.terminate()
        process.terminate.assert_not_called()

    def test_kill_blocked(self) -> None:
        """``kill`` does not reach ``process.kill`` under ``--readonly``."""
        selection, process = _make_selection(readonly=True)
        selection.kill()
        process.kill.assert_not_called()

    def test_interrupt_blocked(self) -> None:
        """``interrupt`` does not reach ``process.send_signal`` under ``--readonly``."""
        selection, process = _make_selection(readonly=True)
        selection.interrupt()
        process.send_signal.assert_not_called()

    def test_send_signal_allowed_when_writable(self) -> None:
        """``send_signal`` reaches the process when ``--readonly`` is off."""
        selection, process = _make_selection(readonly=False)
        selection.send_signal(15)
        process.send_signal.assert_called_once_with(15)

    def test_terminate_allowed_when_writable(self) -> None:
        """``terminate`` reaches the process when ``--readonly`` is off."""
        selection, process = _make_selection(readonly=False)
        selection.terminate()
        process.terminate.assert_called_once_with()

    def test_kill_allowed_when_writable(self) -> None:
        """``kill`` reaches the process when ``--readonly`` is off."""
        selection, process = _make_selection(readonly=False)
        selection.kill()
        process.kill.assert_called_once_with()
