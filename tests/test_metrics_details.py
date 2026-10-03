# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

"""Tests for the process-detail rows of the metrics screen (full command and cwd).

The screen is a curses canvas, so the pure row-formatting helpers and the frame shape
are tested here instead of the rendered terminal output.
"""

import types

import pytest

from nvitop.tui.screens.metrics import (
    ProcessMetricsScreen,
    detail_rows,
    format_cwd_line,
    graph_heights,
    process_cwd,
    wrap_command,
)


# ------------------------------------------------------------------------------
# Command wrapping
# ------------------------------------------------------------------------------


def test_short_command_fits_in_one_row():
    assert wrap_command('python main.py', 40) == ('python main.py', '')


def test_exact_fit_fits_in_one_row():
    command = 'x' * 20
    assert wrap_command(command, 20) == (command, '')


def test_long_command_splits_head_and_tail():
    command = 'python main.py --port 18000'  # 28 columns, so two rows of 20 suffice
    head, tail = wrap_command(command, 20)
    assert head == command[:20]
    assert tail == command[20:]
    assert head + tail == command  # nothing is lost when two rows suffice


def test_very_long_command_marks_the_loss_on_the_second_row():
    command = 'x' * 100
    head, tail = wrap_command(command, 20)
    assert len(head) == 20
    assert len(tail) == 20
    assert tail.endswith('..')  # the truncation is visible instead of silent


def test_wrapping_is_wide_character_aware():
    command = '中文中文abc'  # 8 display columns of wide characters, then 3 narrow ones
    head, tail = wrap_command(command, 8)
    assert head == '中文中文'
    assert tail == 'abc'


def test_degenerate_widths_are_safe():
    assert wrap_command('python main.py', 0) == ('', '')
    assert wrap_command('', 10) == ('', '')


# ------------------------------------------------------------------------------
# cwd row
# ------------------------------------------------------------------------------


def test_cwd_line_without_cwd_is_empty():
    assert format_cwd_line(None) == ''
    assert format_cwd_line('') == ''


def test_cwd_line_with_plain_directory():
    assert format_cwd_line('/home/airroot/octopus-api') == 'CWD: /home/airroot/octopus-api'


def test_cwd_line_names_the_container_view():
    assert format_cwd_line('/workspace', 'vllm-svc') == 'CWD: /workspace (container: vllm-svc)'


# ------------------------------------------------------------------------------
# Working directory: plain process information, independent of the recognition
# ------------------------------------------------------------------------------


def test_process_cwd_reads_the_working_directory():
    class FakeProcess:
        def cwd(self):
            return '/srv/octopus'

    assert process_cwd(FakeProcess()) == '/srv/octopus'


def test_process_cwd_is_none_when_unavailable():
    class GoneProcess:
        def cwd(self):
            raise RuntimeError('the process is gone')

    assert process_cwd(GoneProcess()) is None


def test_process_cwd_does_not_depend_on_the_command_recognition():
    from nvitop.api import recognizer

    class FakeProcess:
        pid = 1234

        def cwd(self):
            return '/srv/octopus'

    recognizer.set_enabled(False)
    assert recognizer.process_facts(FakeProcess()) is None  # the recognition is off
    assert process_cwd(FakeProcess()) == '/srv/octopus'  # the row is still filled


# ------------------------------------------------------------------------------
# The three detail rows drawn on the screen
# ------------------------------------------------------------------------------


def test_detail_rows_for_a_short_command():
    rows = detail_rows(['python', 'main.py'], '/srv/octopus', None, width=100)
    assert rows == ('CMD: python main.py', '', 'CWD: /srv/octopus')


def test_detail_rows_wrap_a_long_command_onto_the_second_row():
    cmdline = ['python', 'main.py', '--port', '18000', '--model', '/models/qwen-tts-base']
    rows = detail_rows(cmdline, '/workspace', 'vllm-svc', width=40)
    assert rows[0].startswith('CMD: python main.py')
    assert rows[1].startswith('     ')
    assert rows[1].strip()
    assert rows[2] == 'CWD: /workspace (container: vllm-svc)'


def test_detail_rows_without_a_working_directory():
    rows = detail_rows(['python', 'main.py'], None, None, width=100)
    assert rows[2] == ''


# ------------------------------------------------------------------------------
# Frame shape: the header block carries three detail rows before the graphs
# ------------------------------------------------------------------------------


def test_frame_has_three_blank_detail_rows_before_the_graphs():
    graph_top = ProcessMetricsScreen.GRAPH_TOP
    stub = types.SimpleNamespace(
        width=80,
        left_width=38,
        right_width=40,
        upper_height=14,
        lower_height=15,
    )
    lines = ProcessMetricsScreen.frame_lines(stub)

    assert lines[0].startswith('╒')
    assert lines[0].endswith('╕')
    assert lines[3].startswith('╞')  # separator under the column-header row
    assert '╤' in lines[graph_top - 1]  # separator between the two graph columns

    blank_row = '│' + ' ' * 78 + '│'
    for row in (graph_top - 4, graph_top - 3, graph_top - 2):
        assert lines[row] == blank_row

    # header block + upper graphs + middle rule + lower graphs + bottom border
    assert len(lines) == graph_top + stub.upper_height + 1 + stub.lower_height + 1


@pytest.mark.parametrize('height', [21, 24, 30, 40, 41, 42, 60])
def test_graph_heights_fill_the_screen_height(height):
    upper, lower = graph_heights(height)
    assert ProcessMetricsScreen.GRAPH_TOP + upper + 1 + lower + 1 == height
