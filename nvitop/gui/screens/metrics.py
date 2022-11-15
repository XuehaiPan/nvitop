# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import itertools
import threading
import time
from collections import OrderedDict

from nvitop.gui.library import (
    HOSTNAME,
    NA,
    SUPERUSER,
    USERCONTEXT,
    USERNAME,
    BufferedHistoryGraph,
    Displayable,
    GpuProcess,
    Selection,
    WideString,
    bytes2human,
    cut_string,
    host,
    wcslen,
)


def get_yticks(history, y_offset):  # pylint: disable=too-many-branches,too-many-locals
    height = history.height
    baseline = history.baseline
    bound = history.bound
    max_bound = history.max_bound
    scale = history.scale
    upsidedown = history.upsidedown

    def p2h_f(p):
        return 0.01 * scale * p * (max_bound - baseline) * (height - 1) / (bound - baseline)

    max_height = height - 2
    percentages = (1, 2, 4, 5, 8, 10, 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000)
    h2p = {}
    p2h = {}
    h2e = {}
    for p in percentages:
        h_f = p2h_f(p)
        p2h[p] = h = int(h_f)
        if h not in h2p:
            if h < max_height:
                h2p[h] = p
                h2e[h] = abs(h_f - h) / p
        elif abs(h_f - h) / p < h2e[h]:
            h2p[h] = p
            h2e[h] = abs(h_f - h) / p
    h2p = sorted(h2p.items())
    ticks = []
    if len(h2p) >= 2:
        (hm1, pm1), (h2, p2) = h2p[-2:]
        if height < 12:
            if h2e[hm1] < h2e[h2]:
                ticks = [(hm1, pm1)]
            else:
                ticks = [(h2, p2)]
        else:
            ticks = [(h2, p2)]
            if p2 % 2 == 0:
                p1 = p2 // 2
                h1 = int(p2h_f(p1))
                p3 = 3 * p1
                h3 = int(p2h_f(p3))
                if p1 >= 3:
                    ticks.append((h1, p1))
                    if h2 < h3 < max_height:
                        ticks.append((h3, p3))
    else:
        ticks = list(h2p)
    if not upsidedown:
        ticks = [(height - 1 - h, p) for h, p in ticks]
    return [(h + y_offset, p) for h, p in ticks]


class ProcessMetricsScreen(Displayable):  # pylint: disable=too-many-instance-attributes
    NAME = 'process-metrics'
    SNAPSHOT_INTERVAL = 0.5

    def __init__(self, win, root):
        super().__init__(win, root)

        self.selection = Selection(panel=self)
        self.used_gpu_memory = None
        self.gpu_sm_utilization = None
        self.cpu_percent = None
        self.used_host_memory = None

        self.enabled = False
        self.snapshot_lock = threading.Lock()
        self._snapshot_daemon = threading.Thread(
            name='process-metrics-snapshot-daemon', target=self._snapshot_target, daemon=True
        )
        self._daemon_running = threading.Event()

        self.x, self.y = root.x, root.y
        self.width, self.height = root.width, root.height
        self.left_width = max(20, (self.width - 3) // 2)
        self.right_width = max(20, (self.width - 2) // 2)
        self.upper_height = max(5, (self.height - 5 - 3) // 2)
        self.lower_height = max(5, (self.height - 5 - 2) // 2)

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        if self._visible != value:
            self.need_redraw = True
            self._visible = value
        if self.visible:
            self._daemon_running.set()
            try:
                self._snapshot_daemon.start()
            except RuntimeError:
                pass
            self.take_snapshots()
        else:
            self.focused = False

    def enable(self, state=True):
        if not self.selection.is_set() or not state:
            self.disable()
            return

        total_host_memory = host.virtual_memory().total
        total_host_memory_human = bytes2human(total_host_memory)
        total_gpu_memory = self.process.device.memory_total()
        total_gpu_memory_human = bytes2human(total_gpu_memory)

        def format_cpu_percent(value):
            if value is NA:
                return 'CPU: {}'.format(value)
            return 'CPU: {:.1f}%'.format(value)

        def format_max_cpu_percent(value):
            if value is NA:
                return 'MAX CPU: {}'.format(value)
            return 'MAX CPU: {:.1f}%'.format(value)

        def format_host_memory(value):
            if value is NA:
                return 'HOST-MEM: {}'.format(value)
            return 'HOST-MEM: {} ({:.1f}%)'.format(
                bytes2human(value),
                round(100.0 * value / total_host_memory, 1),
            )

        def format_max_host_memory(value):
            if value is NA:
                return 'MAX HOST-MEM: {}'.format(value)
            return 'MAX HOST-MEM: {} ({:.1f}%) / {}'.format(
                bytes2human(value),
                round(100.0 * value / total_host_memory, 1),
                total_host_memory_human,
            )

        def format_gpu_memory(value):
            if value is not NA and total_gpu_memory is not NA:
                return 'GPU-MEM: {} ({:.1f}%)'.format(
                    bytes2human(value),
                    round(100.0 * value / total_gpu_memory, 1),
                )
            return 'GPU-MEM: {}'.format(value)

        def format_max_gpu_memory(value):
            if value is not NA and total_gpu_memory is not NA:
                return 'MAX GPU-MEM: {} ({:.1f}%) / {}'.format(
                    bytes2human(value),
                    round(100.0 * value / total_gpu_memory, 1),
                    total_gpu_memory_human,
                )
            return 'MAX GPU-MEM: {}'.format(value)

        def format_sm(value):
            if value is NA:
                return 'GPU-SM: {}'.format(value)
            return 'GPU-SM: {:.1f}%'.format(value)

        def format_max_sm(value):
            if value is NA:
                return 'MAX GPU-SM: {}'.format(value)
            return 'MAX GPU-SM: {:.1f}%'.format(value)

        with self.snapshot_lock:
            self.cpu_percent = BufferedHistoryGraph(
                interval=1.0,
                upperbound=1000.0,
                width=self.left_width,
                height=self.upper_height,
                baseline=0.0,
                upsidedown=False,
                dynamic_bound=True,
                min_bound=10.0,
                init_bound=100.0,
                format=format_cpu_percent,
                max_format=format_max_cpu_percent,
            )
            self.used_host_memory = BufferedHistoryGraph(
                interval=1.0,
                upperbound=total_host_memory,
                width=self.left_width,
                height=self.lower_height,
                baseline=0.0,
                upsidedown=True,
                dynamic_bound=True,
                format=format_host_memory,
                max_format=format_max_host_memory,
            )
            self.used_gpu_memory = BufferedHistoryGraph(
                interval=1.0,
                upperbound=total_gpu_memory or 1,
                width=self.right_width,
                height=self.upper_height,
                baseline=0.0,
                upsidedown=False,
                dynamic_bound=True,
                format=format_gpu_memory,
                max_format=format_max_gpu_memory,
            )
            self.gpu_sm_utilization = BufferedHistoryGraph(
                interval=1.0,
                upperbound=100.0,
                width=self.right_width,
                height=self.lower_height,
                baseline=0.0,
                upsidedown=True,
                dynamic_bound=True,
                format=format_sm,
                max_format=format_max_sm,
            )
            self.cpu_percent.scale = 0.1
            self.used_host_memory.scale = 1.0
            self.used_gpu_memory.scale = 1.0
            self.gpu_sm_utilization.scale = 1.0

            self._daemon_running.set()
            try:
                self._snapshot_daemon.start()
            except RuntimeError:
                pass
            self.enabled = True

        self.take_snapshots()
        self.update_size()

    def disable(self):
        with self.snapshot_lock:
            self._daemon_running.clear()
            self.enabled = False
            self.cpu_percent = None
            self.used_host_memory = None
            self.used_gpu_memory = None
            self.gpu_sm_utilization = None

    @property
    def process(self):
        return self.selection.process

    @process.setter
    def process(self, value):
        self.selection.process = value
        self.enable()

    def take_snapshots(self):
        with self.snapshot_lock:
            if not self.selection.is_set() or not self.enabled:
                return

            with GpuProcess.failsafe():
                self.process.device.as_snapshot()
                self.process.update_gpu_status()
                snapshot = self.process.as_snapshot()

                self.cpu_percent.add(snapshot.cpu_percent)
                self.used_host_memory.add(snapshot.host_memory)
                self.used_gpu_memory.add(snapshot.gpu_memory)
                self.gpu_sm_utilization.add(snapshot.gpu_sm_utilization)

    def _snapshot_target(self):
        while True:
            self._daemon_running.wait()
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def update_size(self, termsize=None):
        n_term_lines, n_term_cols = termsize = super().update_size(termsize=termsize)

        self.width = n_term_cols - self.x
        self.height = n_term_lines - self.y
        self.left_width = max(20, (self.width - 3) // 2)
        self.right_width = max(20, (self.width - 2) // 2)
        self.upper_height = max(5, (self.height - 8) // 2)
        self.lower_height = max(5, (self.height - 7) // 2)
        self.need_redraw = True

        with self.snapshot_lock:
            if self.enabled:
                self.cpu_percent.graph_size = (self.left_width, self.upper_height)
                self.used_host_memory.graph_size = (self.left_width, self.lower_height)
                self.used_gpu_memory.graph_size = (self.right_width, self.upper_height)
                self.gpu_sm_utilization.graph_size = (self.right_width, self.lower_height)

        return termsize

    def frame_lines(self):
        line = '│' + ' ' * self.left_width + '│' + ' ' * self.right_width + '│'
        frame = [
            '╒' + '═' * (self.width - 2) + '╕',
            '│ {} │'.format('Process:'.ljust(self.width - 4)),
            '│ {} │'.format('GPU'.ljust(self.width - 4)),
            '╞' + '═' * (self.width - 2) + '╡',
            '│' + ' ' * (self.width - 2) + '│',
            '╞' + '═' * self.left_width + '╤' + '═' * self.right_width + '╡',
            *([line] * self.upper_height),
            '├' + '─' * self.left_width + '┼' + '─' * self.right_width + '┤',
            *([line] * self.lower_height),
            '╘' + '═' * self.left_width + '╧' + '═' * self.right_width + '╛',
        ]
        return frame

    def poke(self):
        if self.visible and not self._daemon_running.is_set():
            self._daemon_running.set()
            try:
                self._snapshot_daemon.start()
            except RuntimeError:
                pass
            self.take_snapshots()

        super().poke()

    def draw(self):  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
        self.color_reset()

        if self.need_redraw:
            for y, line in enumerate(self.frame_lines(), start=self.y):
                self.addstr(y, self.x, line)

            context_width = wcslen(USERCONTEXT)
            if not host.WINDOWS or len(USERCONTEXT) == context_width:
                # Do not support windows-curses with wide characters
                username_width = wcslen(USERNAME)
                hostname_width = wcslen(HOSTNAME)
                offset = self.x + self.width - context_width - 2
                self.addstr(self.y + 1, self.x + offset, USERCONTEXT)
                self.color_at(self.y + 1, self.x + offset, width=context_width, attr='bold')
                self.color_at(
                    self.y + 1,
                    self.x + offset,
                    width=username_width,
                    fg=('yellow' if SUPERUSER else 'magenta'),
                    attr='bold',
                )
                self.color_at(
                    self.y + 1,
                    self.x + offset + username_width + 1,
                    width=hostname_width,
                    fg='green',
                    attr='bold',
                )

            for offset, string in (
                (19, '╴30s├'),
                (34, '╴60s├'),
                (65, '╴120s├'),
                (95, '╴180s├'),
                (125, '╴240s├'),
                (155, '╴300s├'),
            ):
                for x_offset, width in (
                    (self.x + 1 + self.left_width, self.left_width),
                    (self.x + 1 + self.left_width + 1 + self.right_width, self.right_width),
                ):
                    if offset > width:
                        break
                    self.addstr(self.y + self.upper_height + 6, x_offset - offset, string)
                    self.color_at(
                        self.y + self.upper_height + 6,
                        x_offset - offset + 1,
                        width=len(string) - 2,
                        attr='dim',
                    )

        with self.snapshot_lock:
            process = self.process.snapshot
            columns = OrderedDict(
                [
                    (' GPU', self.process.device.display_index.rjust(4)),
                    ('PID  ', '{} {}'.format(str(process.pid).rjust(3), process.type)),
                    (
                        'USER',
                        WideString(
                            cut_string(
                                WideString(process.username).rjust(4),
                                maxlen=32,
                                padstr='+',
                            )
                        ),
                    ),
                    (' GPU-MEM', process.gpu_memory_human.rjust(8)),
                    (' %SM', str(process.gpu_sm_utilization).rjust(4)),
                    ('%GMBW', str(process.gpu_memory_utilization).rjust(5)),
                    ('%ENC', str(process.gpu_encoder_utilization).rjust(4)),
                    ('%DEC', str(process.gpu_encoder_utilization).rjust(4)),
                    ('  %CPU', process.cpu_percent_string.rjust(6)),
                    (' %MEM', process.memory_percent_string.rjust(5)),
                    (' TIME', (' ' + process.running_time_human).rjust(5)),
                ]
            )

            x = self.x + 1
            header = ''
            fields = WideString()
            no_break = True
            for i, (col, value) in enumerate(columns.items()):
                width = len(value)
                if x + width < self.width - 2:
                    if i == 0:
                        header += col.rjust(width)
                        fields += value
                    else:
                        header += ' ' + col.rjust(width)
                        fields += ' ' + value
                    x = self.x + 1 + len(fields)
                else:
                    no_break = False
                    break

            self.addstr(self.y + 2, self.x + 1, header.ljust(self.width - 2))
            self.addstr(self.y + 4, self.x + 1, str(fields.ljust(self.width - 2)))
            self.color_at(
                self.y + 4, self.x + 1, width=4, fg=self.process.device.snapshot.display_color
            )

            if no_break:
                x = self.x + 1 + len(fields) + 2
                if x + 4 < self.width - 2:
                    self.addstr(
                        self.y + 2,
                        x,
                        cut_string('COMMAND', self.width - x - 2, padstr='..').ljust(
                            self.width - x - 2
                        ),
                    )
                    if process.is_zombie or process.no_permissions:
                        self.color(fg='yellow')
                    elif process.is_gone:
                        self.color(fg='red')
                    self.addstr(
                        self.y + 4,
                        x,
                        cut_string(
                            WideString(process.command).ljust(self.width - x - 2),
                            self.width - x - 2,
                            padstr='..',
                        ),
                    )

            self.color(fg='cyan')
            for y, line in enumerate(self.cpu_percent.graph, start=self.y + 6):
                self.addstr(y, self.x + 1, line)

            self.color(fg='magenta')
            for y, line in enumerate(
                self.used_host_memory.graph, start=self.y + self.upper_height + 7
            ):
                self.addstr(y, self.x + 1, line)

            if self.TERM_256COLOR:
                scale = (self.used_gpu_memory.bound / self.used_gpu_memory.max_bound) / (
                    self.upper_height - 1
                )
                for i, (y, line) in enumerate(
                    enumerate(self.used_gpu_memory.graph, start=self.y + 6)
                ):
                    self.addstr(
                        y,
                        self.x + self.left_width + 2,
                        line,
                        self.get_fg_bg_attr(fg=(self.upper_height - i - 1) * scale),
                    )

                scale = (self.gpu_sm_utilization.bound / self.gpu_sm_utilization.max_bound) / (
                    self.lower_height - 1
                )
                for i, (y, line) in enumerate(
                    enumerate(self.gpu_sm_utilization.graph, start=self.y + self.upper_height + 7)
                ):
                    self.addstr(
                        y,
                        self.x + self.left_width + 2,
                        line,
                        self.get_fg_bg_attr(fg=i * scale),
                    )
            else:
                self.color(fg=self.process.device.snapshot.memory_display_color)
                for y, line in enumerate(self.used_gpu_memory.graph, start=self.y + 6):
                    self.addstr(y, self.x + self.left_width + 2, line)

                self.color(fg=self.process.device.snapshot.gpu_display_color)
                for y, line in enumerate(
                    self.gpu_sm_utilization.graph, start=self.y + self.upper_height + 7
                ):
                    self.addstr(y, self.x + self.left_width + 2, line)

            self.color_reset()
            self.addstr(self.y + 6, self.x + 1, ' {} '.format(self.cpu_percent.max_value_string()))
            self.addstr(self.y + 7, self.x + 5, ' {} '.format(self.cpu_percent))
            self.addstr(
                self.y + self.upper_height + self.lower_height + 5,
                self.x + 5,
                ' {} '.format(self.used_host_memory),
            )
            self.addstr(
                self.y + self.upper_height + self.lower_height + 6,
                self.x + 1,
                ' {} '.format(
                    cut_string(
                        self.used_host_memory.max_value_string(),
                        maxlen=self.left_width - 2,
                        padstr='..',
                    )
                ),
            )
            self.addstr(
                self.y + 6,
                self.x + self.left_width + 2,
                ' {} '.format(
                    cut_string(
                        self.used_gpu_memory.max_value_string(),
                        maxlen=self.right_width - 2,
                        padstr='..',
                    )
                ),
            )
            self.addstr(
                self.y + 7, self.x + self.left_width + 6, ' {} '.format(self.used_gpu_memory)
            )
            self.addstr(
                self.y + self.upper_height + self.lower_height + 5,
                self.x + self.left_width + 6,
                ' {} '.format(self.gpu_sm_utilization),
            )
            self.addstr(
                self.y + self.upper_height + self.lower_height + 6,
                self.x + self.left_width + 2,
                ' {} '.format(self.gpu_sm_utilization.max_value_string()),
            )

            for y in range(self.y + 6, self.y + 6 + self.upper_height):
                self.addstr(y, self.x, '│')
                self.addstr(y, self.x + self.left_width + 1, '│')
            for y in range(
                self.y + self.upper_height + 7, self.y + self.upper_height + self.lower_height + 7
            ):
                self.addstr(y, self.x, '│')
                self.addstr(y, self.x + self.left_width + 1, '│')

            self.color(attr='dim')
            for y, p in itertools.chain(
                get_yticks(self.cpu_percent, self.y + 6),
                get_yticks(self.used_host_memory, self.y + self.upper_height + 7),
            ):
                self.addstr(y, self.x, '├╴{}% '.format(p))
                self.color_at(y, self.x, width=2, attr=0)
            x = self.x + self.left_width + 1
            for y, p in itertools.chain(
                get_yticks(self.used_gpu_memory, self.y + 6),
                get_yticks(self.gpu_sm_utilization, self.y + self.upper_height + 7),
            ):
                self.addstr(y, x, '├╴{}% '.format(p))
                self.color_at(y, x, width=2, attr=0)

    def destroy(self):
        super().destroy()
        self._daemon_running.clear()

    def press(self, key):
        self.root.keymaps.use_keymap('process-metrics')
        self.root.press(key)
