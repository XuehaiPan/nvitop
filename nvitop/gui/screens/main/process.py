# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import itertools
import threading
import time
from collections import namedtuple
from operator import attrgetter, xor

from cachetools.func import ttl_cache

from nvitop.gui.library import (
    HOSTNAME,
    LARGE_INTEGER,
    SUPERUSER,
    USERCONTEXT,
    USERNAME,
    Displayable,
    GpuProcess,
    MouseEvent,
    Selection,
    WideString,
    colored,
    cut_string,
    host,
    wcslen,
)


Order = namedtuple('Order', ['key', 'reverse', 'offset', 'column', 'previous', 'next'])


class ProcessPanel(Displayable):  # pylint: disable=too-many-instance-attributes
    NAME = 'process'
    SNAPSHOT_INTERVAL = 0.67

    ORDERS = {
        'natural': Order(
            key=attrgetter('device.tuple_index', '_gone', 'username', 'pid'),
            reverse=False,
            offset=3,
            column='ID',
            previous='time',
            next='pid',
        ),
        'pid': Order(
            key=attrgetter('_gone', 'pid', 'device.tuple_index'),
            reverse=False,
            offset=10,
            column='PID',
            previous='natural',
            next='username',
        ),
        'username': Order(
            key=attrgetter('_gone', 'username', 'pid', 'device.tuple_index'),
            reverse=False,
            offset=19,
            column='USER',
            previous='pid',
            next='gpu_memory',
        ),
        'gpu_memory': Order(
            key=attrgetter(
                '_gone',
                'gpu_memory',
                'gpu_sm_utilization',
                'cpu_percent',
                'pid',
                'device.tuple_index',
            ),
            reverse=True,
            offset=25,
            column='GPU-MEM',
            previous='username',
            next='sm_utilization',
        ),
        'sm_utilization': Order(
            key=attrgetter(
                '_gone',
                'gpu_sm_utilization',
                'gpu_memory',
                'cpu_percent',
                'pid',
                'device.tuple_index',
            ),
            reverse=True,
            offset=34,
            column='SM',
            previous='gpu_memory',
            next='cpu_percent',
        ),
        'cpu_percent': Order(
            key=attrgetter('_gone', 'cpu_percent', 'memory_percent', 'pid', 'device.tuple_index'),
            reverse=True,
            offset=38,
            column='%CPU',
            previous='sm_utilization',
            next='memory_percent',
        ),
        'memory_percent': Order(
            key=attrgetter('_gone', 'memory_percent', 'cpu_percent', 'pid', 'device.tuple_index'),
            reverse=True,
            offset=44,
            column='%MEM',
            previous='cpu_percent',
            next='time',
        ),
        'time': Order(
            key=attrgetter('_gone', 'running_time', 'pid', 'device.tuple_index'),
            reverse=True,
            offset=50,
            column='TIME',
            previous='memory_percent',
            next='natural',
        ),
    }

    def __init__(self, devices, compact, filters, win, root):  # pylint: disable=too-many-arguments
        super().__init__(win, root)

        self.devices = devices

        self._compact = compact
        self.width = max(79, root.width)
        self.height = self._full_height = self.compact_height = 7

        self.filters = [None, *filters]

        self.host_headers = ['%CPU', '%MEM', 'TIME', 'COMMAND']

        self.selection = Selection(panel=self)
        self.host_offset = -1
        self.y_mouse = None

        self._order = 'natural'
        self.reverse = False

        self.has_snapshots = False
        self._snapshot_buffer = None
        self._snapshots = []
        self.snapshot_lock = threading.Lock()
        self._snapshot_daemon = threading.Thread(
            name='process-snapshot-daemon', target=self._snapshot_target, daemon=True
        )
        self._daemon_running = threading.Event()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        width = max(79, value)
        if self._width != width and self.visible:
            self.need_redraw = True
        self._width = width

    @property
    def compact(self):
        return self._compact or self.order != 'natural'

    @compact.setter
    def compact(self, value):
        if self._compact != value:
            self.need_redraw = True
            self._compact = value
            processes = self.snapshots
            n_processes, n_devices = len(processes), len(
                set(p.device.physical_index for p in processes)
            )
            self.full_height = 1 + max(6, 5 + n_processes + n_devices - 1)
            self.compact_height = 1 + max(6, 5 + n_processes)
            self.height = self.compact_height if self.compact else self.full_height

    @property
    def full_height(self):
        return self._full_height if self.order == 'natural' else self.compact_height

    @full_height.setter
    def full_height(self, value):
        self._full_height = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if self._order != value:
            self._order = value
            self.height = self.compact_height if self.compact else self.full_height

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots):
        if snapshots is None:
            return
        self.has_snapshots = True

        time_length = max(4, max((len(p.running_time_human) for p in snapshots), default=4))
        time_header = ' ' * (time_length - 4) + 'TIME'
        info_length = max((len(p.host_info) for p in snapshots), default=0)
        n_processes, n_devices = len(snapshots), len(
            set(p.device.physical_index for p in snapshots)
        )
        self.full_height = 1 + max(6, 5 + n_processes + n_devices - 1)
        self.compact_height = 1 + max(6, 5 + n_processes)
        height = self.compact_height if self.compact else self.full_height
        key, reverse, *_ = self.ORDERS[self.order]
        snapshots.sort(key=key, reverse=xor(reverse, self.reverse))

        old_host_offset = self.host_offset
        with self.snapshot_lock:
            self.need_redraw = (
                self.need_redraw or self.height > height or self.host_headers[-2] != time_header
            )
            self._snapshots = snapshots

            self.host_headers[-2] = time_header
            self.height = height
            self.host_offset = max(-1, min(self.host_offset, info_length - self.width + 39))

        if old_host_offset not in (self.host_offset, LARGE_INTEGER):
            self.beep()

        if self.selection.is_set():
            identity = self.selection.identity
            self.selection.reset()
            for i, process in enumerate(snapshots):
                if process._ident == identity:  # pylint: disable=protected-access
                    self.selection.index = i
                    self.selection.process = process
                    break

    @classmethod
    def set_snapshot_interval(cls, interval):
        assert interval > 0.0
        interval = float(interval)

        cls.SNAPSHOT_INTERVAL = min(interval / 3.0, 1.0)
        cls.take_snapshots = ttl_cache(ttl=interval)(
            cls.take_snapshots.__wrapped__  # pylint: disable=no-member
        )

    def ensure_snapshots(self):
        if not self.has_snapshots:
            self.snapshots = self.take_snapshots()

    @ttl_cache(ttl=2.0)
    def take_snapshots(self):
        snapshots = GpuProcess.take_snapshots(self.processes, failsafe=True)
        for condition in self.filters:
            snapshots = filter(condition, snapshots)
        snapshots = list(snapshots)

        time_length = max(4, max((len(p.running_time_human) for p in snapshots), default=4))
        for snapshot in snapshots:
            snapshot.host_info = WideString(
                '{:>5} {:>5}  {}  {}'.format(
                    snapshot.cpu_percent_string.replace('%', ''),
                    snapshot.memory_percent_string.replace('%', ''),
                    ' ' * (time_length - len(snapshot.running_time_human))
                    + snapshot.running_time_human,
                    snapshot.command,
                )
            )

        with self.snapshot_lock:
            self._snapshot_buffer = snapshots

        return snapshots

    def _snapshot_target(self):
        self._daemon_running.wait()
        while self._daemon_running.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def header_lines(self):
        header = [
            '╒' + '═' * (self.width - 2) + '╕',
            '│ {} │'.format('Processes:'.ljust(self.width - 4)),
            '│ GPU     PID      USER  GPU-MEM %SM  {} │'.format(
                '  '.join(self.host_headers).ljust(self.width - 40)
            ),
            '╞' + '═' * (self.width - 2) + '╡',
        ]
        if len(self.snapshots) == 0:
            if self.has_snapshots:
                message = ' No running processes found{} '.format(' (in WSL)' if host.WSL else '')
            else:
                message = ' Gathering process status...'
            header.extend(
                ['│ {} │'.format(message.ljust(self.width - 4)), '╘' + '═' * (self.width - 2) + '╛']
            )
        return header

    @property
    def processes(self):
        return list(
            itertools.chain.from_iterable(device.processes().values() for device in self.devices)
        )

    def poke(self):
        if not self._daemon_running.is_set():
            self._daemon_running.set()
            self._snapshot_daemon.start()

        self.snapshots = self._snapshot_buffer

        self.selection.within_window = False
        if len(self.snapshots) > 0 and self.selection.is_set():
            y = self.y + 5
            prev_device_index = None
            for process in self.snapshots:
                device_index = process.device.physical_index
                if prev_device_index != device_index:
                    if not self.compact and prev_device_index is not None:
                        y += 1
                    prev_device_index = device_index

                if self.selection.is_same(process):
                    self.selection.within_window = (
                        self.root.y <= y < self.root.termsize[0] and self.width >= 79
                    )
                    if not self.selection.within_window:
                        if y < self.root.y:
                            self.parent.y += self.root.y - y
                        elif y >= self.root.termsize[0]:
                            self.parent.y -= y - self.root.termsize[0] + 1
                        self.parent.update_size(self.root.termsize)
                        self.need_redraw = True
                    break
                y += 1

        super().poke()

    def draw(self):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self.color_reset()

        if self.need_redraw:
            if SUPERUSER:
                self.addstr(self.y, self.x + 1, '!CAUTION: SUPERUSER LOGGED-IN.')
                self.color_at(self.y, self.x + 1, width=1, fg='red', attr='blink')
                self.color_at(self.y, self.x + 2, width=29, fg='yellow', attr='italic')

            for y, line in enumerate(self.header_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)

            context_width = wcslen(USERCONTEXT)
            if not host.WINDOWS or len(USERCONTEXT) == context_width:
                # Do not support windows-curses with wide characters
                username_width = wcslen(USERNAME)
                hostname_width = wcslen(HOSTNAME)
                offset = self.x + self.width - context_width - 2
                self.addstr(self.y + 2, self.x + offset, USERCONTEXT)
                self.color_at(self.y + 2, self.x + offset, width=context_width, attr='bold')
                self.color_at(
                    self.y + 2,
                    self.x + offset,
                    width=username_width,
                    fg=('yellow' if SUPERUSER else 'magenta'),
                    attr='bold',
                )
                self.color_at(
                    self.y + 2,
                    self.x + offset + username_width + 1,
                    width=hostname_width,
                    fg='green',
                    attr='bold',
                )

        self.addstr(self.y + 3, self.x + 1, ' GPU     PID      USER  GPU-MEM %SM  ')
        host_offset = max(self.host_offset, 0)
        command_offset = max(14 + len(self.host_headers[-2]) - host_offset, 0)
        if command_offset > 0:
            host_headers = '  '.join(self.host_headers)
            self.addstr(
                self.y + 3,
                self.x + 38,
                '{}'.format(host_headers[host_offset:].ljust(self.width - 40)),
            )
        else:
            self.addstr(self.y + 3, self.x + 38, '{}'.format('COMMAND'.ljust(self.width - 40)))

        _, reverse, offset, column, *_ = self.ORDERS[self.order]
        column_width = len(column)
        reverse = xor(reverse, self.reverse)
        indicator = '▼' if reverse else '▲'
        if self.order in ('cpu_percent', 'memory_percent', 'time'):
            offset -= host_offset
            if self.order == 'time':
                offset += len(self.host_headers[-2]) - 4
            if offset > 38 or host_offset == 0:
                self.addstr(self.y + 3, self.x + offset - 1, column + indicator)
                self.color_at(
                    self.y + 3, self.x + offset - 1, width=column_width, attr='bold | underline'
                )
            elif offset <= 38 < offset + column_width:
                self.addstr(self.y + 3, self.x + 38, (column + indicator)[39 - offset :])
                if offset + column_width >= 40:
                    self.color_at(
                        self.y + 3,
                        self.x + 38,
                        width=offset + column_width - 39,
                        attr='bold | underline',
                    )
            if offset + column_width >= 39:
                self.color_at(self.y + 3, self.x + offset + column_width - 1, width=1, attr='bold')
        elif self.order == 'natural' and not reverse:
            self.color_at(self.y + 3, self.x + 2, width=3, attr='bold')
        else:
            self.addstr(self.y + 3, self.x + offset - 1, column + indicator)
            self.color_at(
                self.y + 3, self.x + offset - 1, width=column_width, attr='bold | underline'
            )
            self.color_at(self.y + 3, self.x + offset + column_width - 1, width=1, attr='bold')

        hint = True
        if self.y_mouse is not None:
            self.selection.reset()
            hint = False

        self.selection.within_window = False
        if len(self.snapshots) > 0:
            y = self.y + 5
            prev_device_index = None
            prev_device_display_index = None
            color = -1
            for process in self.snapshots:
                device_index = process.device.physical_index
                device_display_index = process.device.display_index
                if prev_device_index != device_index:
                    if not self.compact and prev_device_index is not None:
                        self.addstr(y, self.x, '├' + '─' * (self.width - 2) + '┤')
                        if y == self.y_mouse:
                            self.y_mouse += 1
                        y += 1
                    prev_device_index = device_index
                if prev_device_display_index != device_display_index:
                    color = process.device.snapshot.display_color
                    prev_device_display_index = device_display_index

                host_info = process.host_info
                if self.host_offset < 0:
                    host_info = cut_string(host_info, padstr='..', maxlen=self.width - 39)
                else:
                    host_info = WideString(host_info)[self.host_offset :]
                self.addstr(
                    y,
                    self.x,
                    '│{:>4} {:>7} {} {} {:>8} {:>3} {} │'.format(
                        device_display_index,
                        cut_string(process.pid, maxlen=7, padstr='.'),
                        process.type,
                        str(
                            WideString(cut_string(process.username, maxlen=7, padstr='+')).rjust(7)
                        ),
                        process.gpu_memory_human,
                        process.gpu_sm_utilization_string.replace('%', ''),
                        WideString(host_info).ljust(self.width - 39)[: self.width - 39],
                    ),
                )
                if self.host_offset > 0:
                    self.addstr(y, self.x + 37, ' ')

                is_zombie = process.is_zombie
                no_permissions = process.no_permissions
                is_gone = process.is_gone
                if (is_zombie or no_permissions or is_gone) and command_offset == 0:
                    self.addstr(y, self.x + 38, process.command)

                if y == self.y_mouse:
                    self.selection.process = process
                    hint = True

                if self.selection.is_same(process):
                    self.color_at(
                        y,
                        self.x + 1,
                        width=self.width - 2,
                        fg='yellow' if self.selection.is_tagged(process) else 'cyan',
                        attr='bold | reverse',
                    )
                    self.selection.within_window = (
                        self.root.y <= y < self.root.termsize[0] and self.width >= 79
                    )
                else:
                    owned = str(process.username) == USERNAME or SUPERUSER
                    if self.selection.is_same_on_host(process):
                        self.addstr(y, self.x + 1, '=', self.get_fg_bg_attr(attr='bold | blink'))
                    self.color_at(y, self.x + 2, width=3, fg=color)
                    if self.selection.is_tagged(process):
                        self.color_at(
                            y,
                            self.x + 5,
                            width=self.width - 6,
                            fg='yellow',
                            attr='bold' if owned else 'bold | dim',
                        )
                    elif not owned:
                        self.color_at(y, self.x + 5, width=self.width - 6, attr='dim')
                    if is_zombie or no_permissions:
                        self.color_at(y, self.x + 38 + command_offset, width=14, fg='yellow')
                    elif is_gone:
                        self.color_at(y, self.x + 38 + command_offset, width=15, fg='red')
                y += 1

            self.addstr(y, self.x, '╘' + '═' * (self.width - 2) + '╛')
            if not hint:
                self.selection.clear()

        elif self.has_snapshots:
            message = ' No running processes found{} '.format(' (in WSL)' if host.WSL else '')
            self.addstr(self.y + 5, self.x, '│ {} │'.format(message.ljust(self.width - 4)))

        text_offset = self.x + self.width - 47
        if len(self.selection.tagged) > 0 or (
            self.selection.owned() and self.selection.within_window
        ):
            self.addstr(self.y, text_offset, '(Press ^C(INT)/T(TERM)/K(KILL) to send signals)')
            self.color_at(self.y, text_offset + 7, width=2, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 10, width=3, fg='red', attr='bold')
            self.color_at(self.y, text_offset + 15, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 17, width=4, fg='red', attr='bold')
            self.color_at(self.y, text_offset + 23, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 25, width=4, fg='red', attr='bold')
        else:
            self.addstr(self.y, text_offset, ' ' * 47)

    def finalize(self):
        self.y_mouse = None
        super().finalize()

    def destroy(self):
        super().destroy()
        self._daemon_running.clear()

    def print_width(self):
        self.ensure_snapshots()
        return min(
            self.width, max((39 + len(process.host_info) for process in self.snapshots), default=79)
        )

    def print(self):
        self.ensure_snapshots()

        lines = ['', *self.header_lines()]
        lines[2] = ''.join(
            (
                lines[2][: -2 - wcslen(USERCONTEXT)],
                colored(USERNAME, color=('yellow' if SUPERUSER else 'magenta'), attrs=('bold',)),
                colored('@', attrs=('bold',)),
                colored(HOSTNAME, color='green', attrs=('bold',)),
                lines[2][-2:],
            )
        )

        if len(self.snapshots) > 0:
            key, reverse, *_ = self.ORDERS['natural']
            self.snapshots.sort(key=key, reverse=reverse)
            prev_device_index = None
            prev_device_display_index = None
            color = None
            for process in self.snapshots:
                device_index = process.device.physical_index
                device_display_index = process.device.display_index
                if prev_device_index != device_index:
                    if prev_device_index is not None:
                        lines.append('├' + '─' * (self.width - 2) + '┤')
                    prev_device_index = device_index
                if prev_device_display_index != device_display_index:
                    color = process.device.snapshot.display_color
                    prev_device_display_index = device_display_index

                host_info = cut_string(process.host_info, padstr='..', maxlen=self.width - 39)

                info = '{:>7} {} {} {:>8} {:>3} {}'.format(
                    cut_string(process.pid, maxlen=7, padstr='.'),
                    process.type,
                    str(WideString(cut_string(process.username, maxlen=7, padstr='+')).rjust(7)),
                    process.gpu_memory_human,
                    process.gpu_sm_utilization_string.replace('%', ''),
                    WideString(host_info).ljust(self.width - 39)[: self.width - 39],
                )
                if process.is_zombie or process.no_permissions or process.is_gone:
                    info = info.split(process.command)
                    if process.username != USERNAME and not SUPERUSER:
                        info = map(lambda item: colored(item, attrs=('dark',)), info)
                    info = colored(
                        process.command, color=('red' if process.is_gone else 'yellow')
                    ).join(info)
                elif process.username != USERNAME and not SUPERUSER:
                    info = colored(info, attrs=('dark',))
                lines.append(
                    '│{} {} │'.format(colored('{:>4}'.format(device_display_index), color), info)
                )

            lines.append('╘' + '═' * (self.width - 2) + '╛')

        lines = '\n'.join(lines)
        if self.ascii:
            lines = lines.translate(self.ASCII_TRANSTABLE)

        try:
            print(lines)
        except UnicodeError:
            print(lines.translate(self.ASCII_TRANSTABLE))

    def press(self, key):
        self.root.keymaps.use_keymap('process')
        self.root.press(key)

    def click(self, event):
        if event.pressed(1) or event.pressed(3) or event.clicked(1) or event.clicked(3):
            self.y_mouse = event.y
            return True

        direction = event.wheel_direction()
        if event.shift():
            self.host_offset += 2 * direction
        else:
            self.selection.move(direction=direction)
        return True

    def __contains__(self, item):
        if self.parent.visible and isinstance(item, MouseEvent):
            return True
        return super().__contains__(item)
