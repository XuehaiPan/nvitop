# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import threading
import time
from collections import OrderedDict
from operator import attrgetter, xor

from cachetools.func import ttl_cache

from nvitop.core import NA, host, GpuProcess
from nvitop.gui.library import (Displayable, MouseEvent, colored, cut_string,
                                CURRENT_USER, IS_SUPERUSER)
from nvitop.gui.screens.main.utils import Order, Selected


class ProcessPanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7
    ORDERS = {
        'natural': Order(
            key=attrgetter('device.index', '_gone', 'username', 'pid'),
            reverse=False, offset=3, column='ID', previous='time', next='pid'
        ),
        'pid': Order(
            key=attrgetter('_gone', 'pid', 'device.index'),
            reverse=False, offset=10, column='PID', previous='natural', next='username'
        ),
        'username': Order(
            key=attrgetter('_gone', 'username', 'pid', 'device.index'),
            reverse=False, offset=19, column='USER', previous='pid', next='gpu_memory'
        ),
        'gpu_memory': Order(
            key=attrgetter('_gone', 'gpu_memory', 'gpu_sm_utilization', 'cpu_percent', 'pid', 'device.index'),
            reverse=True, offset=25, column='GPU-MEM', previous='username', next='sm_utilization'
        ),
        'sm_utilization': Order(
            key=attrgetter('_gone', 'gpu_sm_utilization', 'gpu_memory', 'cpu_percent', 'pid', 'device.index'),
            reverse=True, offset=34, column='SM', previous='gpu_memory', next='cpu_percent'
        ),
        'cpu_percent': Order(
            key=attrgetter('_gone', 'cpu_percent', 'memory_percent', 'pid', 'device.index'),
            reverse=True, offset=38, column='%CPU', previous='sm_utilization', next='memory_percent'
        ),
        'memory_percent': Order(
            key=attrgetter('_gone', 'memory_percent', 'cpu_percent', 'pid', 'device.index'),
            reverse=True, offset=44, column='%MEM', previous='cpu_percent', next='time'
        ),
        'time': Order(
            key=attrgetter('_gone', 'running_time', 'pid', 'device.index'),
            reverse=True, offset=50, column='TIME', previous='memory_percent', next='natural'
        ),
    }

    def __init__(self, devices, compact, filters, win, root):
        super().__init__(win, root)

        self.devices = devices
        GpuProcess.CLIENT_MODE = True

        self._compact = compact
        self.width = max(79, root.width)
        self.height = self._full_height = self.compact_height = 7

        self.filters = [None, *filters]

        self.host_headers = ['%CPU', '%MEM', 'TIME', 'COMMAND']

        self.selected = Selected(panel=self)
        self.host_offset = -1

        self._order = 'natural'
        self.reverse = False

        self._snapshot_buffer = []
        self._snapshots = []
        self.snapshot_lock = root.lock
        self.snapshots = self.take_snapshots()
        self._snapshot_daemon = threading.Thread(name='process-snapshot-daemon',
                                                 target=self._snapshot_target, daemon=True)
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
            n_processes, n_devices = len(processes), len(set(p.device.index for p in processes))
            self.full_height = 1 + max(6, 5 + n_processes + n_devices - 1)
            self.compact_height = 1 + max(6, 5 + n_processes)
            self.height = (self.compact_height if self.compact else self.full_height)

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
            self.height = (self.compact_height if self.compact else self.full_height)

    @property
    def snapshots(self):
        return self._snapshots

    @snapshots.setter
    def snapshots(self, snapshots):
        time_length = max(4, max([len(p.running_time_human) for p in snapshots], default=4))
        time_header = ' ' * (time_length - 4) + 'TIME'
        info_length = max([len(p.host_info) for p in snapshots], default=0)
        n_processes, n_devices = len(snapshots), len(set(p.device.index for p in snapshots))
        self.full_height = 1 + max(6, 5 + n_processes + n_devices - 1)
        self.compact_height = 1 + max(6, 5 + n_processes)
        height = (self.compact_height if self.compact else self.full_height)
        key, reverse, *_ = self.ORDERS[self.order]
        snapshots.sort(key=key, reverse=xor(reverse, self.reverse))

        with self.snapshot_lock:
            self._snapshots = snapshots
            self.need_redraw = (self.need_redraw or self.height > height or self.host_headers[-2] != time_header)
            self.host_headers[-2] = time_header
            self.height = height
            old_host_offset = self.host_offset
            self.host_offset = max(-1, min(self.host_offset, info_length - self.width + 39))
            if old_host_offset not in (self.host_offset, 1024):
                self.beep()

            if self.selected.is_set():
                identity = self.selected.identity
                self.selected.clear()
                for i, process in enumerate(snapshots):
                    if process._ident == identity:  # pylint: disable=protected-access
                        self.selected.index = i
                        self.selected.process = process
                        break

    @ttl_cache(ttl=2.0)
    def take_snapshots(self):
        GpuProcess.clear_host_snapshots()
        snapshots = map(GpuProcess.as_snapshot, self.processes.values())
        for condition in self.filters:
            snapshots = filter(condition, snapshots)
        snapshots = list(snapshots)

        time_length = max(4, max([len(p.running_time_human) for p in snapshots], default=4))
        for snapshot in snapshots:
            snapshot.type = snapshot.type.replace('C+G', 'X')
            snapshot.host_info = '{:>5} {:>5}  {}  {}'.format(
                snapshot.cpu_percent_string[:-1],
                snapshot.memory_percent_string[:-1],
                ' ' * (time_length - len(snapshot.running_time_human)) + snapshot.running_time_human,
                snapshot.command
            )
            if snapshot.gpu_memory_human is NA and host.WINDOWS:
                snapshot.gpu_memory_human = 'WDDM:N/A'

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
            '│ GPU     PID      USER  GPU-MEM %SM  {} │'.format('  '.join(self.host_headers).ljust(self.width - 40)),
            '╞' + '═' * (self.width - 2) + '╡',
        ]
        if len(self.snapshots) == 0:
            header.extend([
                '│ {} │'.format(' No running processes found '.ljust(self.width - 4)),
                '╘' + '═' * (self.width - 2) + '╛',
            ])
        return header

    @property
    @ttl_cache(ttl=1.0)
    def processes(self):
        processes = {}
        for device in self.devices:
            for p in device.processes().values():
                processes[(p.device.index, p._gone, p.username(), p.pid)] = p  # pylint: disable=protected-access
        return OrderedDict([((key[-1], key[0]), processes[key]) for key in sorted(processes.keys())])

    def poke(self):
        if not self._daemon_running.is_set():
            self._daemon_running.set()
            self._snapshot_daemon.start()

        with self.snapshot_lock:
            self.snapshots = self._snapshot_buffer

        super().poke()

    def draw(self):
        self.color_reset()

        if self.need_redraw:
            if IS_SUPERUSER:
                self.addstr(self.y, self.x + 1, '!CAUTION: SUPERUSER LOGGED-IN.')
                self.color_at(self.y, self.x + 1, width=1, fg='red', attr='blink')
                self.color_at(self.y, self.x + 2, width=29, fg='yellow', attr='italic')

            for y, line in enumerate(self.header_lines(), start=self.y + 1):
                self.addstr(y, self.x, line)

        self.addstr(self.y + 3, self.x + 1, ' GPU     PID      USER  GPU-MEM %SM  ')
        host_offset = max(self.host_offset, 0)
        command_offset = max(14 + len(self.host_headers[-2]) - host_offset, 0)
        if command_offset > 0:
            host_headers = '  '.join(self.host_headers)
            self.addstr(self.y + 3, self.x + 38, '{}'.format(host_headers[host_offset:].ljust(self.width - 40)))
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
                self.color_at(self.y + 3, self.x + offset - 1, width=column_width, attr='bold | underline')
            elif offset <= 38 < offset + column_width:
                self.addstr(self.y + 3, self.x + 38, (column + indicator)[39 - offset:])
                if offset + column_width >= 40:
                    self.color_at(self.y + 3, self.x + 38, width=offset + column_width - 39, attr='bold | underline')
            if offset + column_width >= 39:
                self.color_at(self.y + 3, self.x + offset + column_width - 1, width=1, attr='bold')
        elif self.order == 'natural' and not reverse:
            self.color_at(self.y + 3, self.x + 2, width=3, attr='bold')
        else:
            self.addstr(self.y + 3, self.x + offset - 1, column + indicator)
            self.color_at(self.y + 3, self.x + offset - 1, width=column_width, attr='bold | underline')
            self.color_at(self.y + 3, self.x + offset + column_width - 1, width=1, attr='bold')

        self.selected.within_window = False
        if len(self.snapshots) > 0:
            y = self.y + 5
            prev_device_index = None
            color = -1
            for process in self.snapshots:
                device_index = process.device.index
                if prev_device_index != device_index:
                    color = process.device.snapshot.display_color
                    if not self.compact and prev_device_index is not None:
                        self.addstr(y, self.x, '├' + '─' * (self.width - 2) + '┤')
                        y += 1
                    prev_device_index = device_index

                host_info = process.host_info
                if self.host_offset < 0:
                    host_info = cut_string(host_info, padstr='..', maxlen=self.width - 39)
                else:
                    host_info = host_info[self.host_offset:self.host_offset + self.width - 39]
                self.addstr(y, self.x,
                            '│ {:>3} {:>7} {} {:>7} {:>8} {:>3} {} │'.format(
                                device_index, cut_string(process.pid, maxlen=7, padstr='.'),
                                process.type, cut_string(process.username, maxlen=7, padstr='+'),
                                process.gpu_memory_human, process.gpu_sm_utilization_string.replace('%', ''),
                                host_info.ljust(self.width - 39)
                            ))
                if self.host_offset > 0:
                    self.addstr(y, self.x + 37, ' ')

                is_zombie = (process.is_running and process.cmdline == ['Zombie Process'])
                no_permissions = (process.is_running and process.cmdline == ['No Permissions'])
                is_gone = (not process.is_running and process.cmdline == ['No Such Process'])
                if (is_zombie or no_permissions or is_gone) and command_offset == 0:
                    self.addstr(y, self.x + 38, process.command)

                if self.selected.is_same(process):
                    self.color_at(y, self.x + 1, width=self.width - 2, fg='cyan', attr='bold | reverse')
                    self.selected.within_window = (0 <= y < self.root.termsize[0] and self.width >= 79)
                else:
                    if self.selected.is_same_on_host(process):
                        self.addstr(y, self.x + 1, '=')
                        self.color_at(y, self.x + 1, width=1, attr='bold | blink')
                    self.color_at(y, self.x + 2, width=3, fg=color)
                    if process.username != CURRENT_USER and not IS_SUPERUSER:
                        self.color_at(y, self.x + 5, width=self.width - 6, attr='dim')
                    if is_zombie or no_permissions:
                        self.color_at(y, self.x + 38 + command_offset, width=14, fg='yellow')
                    elif is_gone:
                        self.color_at(y, self.x + 38 + command_offset, width=15, fg='red')
                y += 1
            self.addstr(y, self.x, '╘' + '═' * (self.width - 2) + '╛')
        else:
            self.addstr(self.y + 5, self.x, '│ {} │'.format(' No running processes found '.ljust(self.width - 4)))

        text_offset = self.x + self.width - 47
        if self.selected.owned() and self.selected.within_window:
            self.addstr(self.y, text_offset, '(Press ^C(INT)/T(TERM)/K(KILL) to send signals)')
            self.color_at(self.y, text_offset + 7, width=2, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 10, width=3, fg='red', attr='bold')
            self.color_at(self.y, text_offset + 15, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 17, width=4, fg='red', attr='bold')
            self.color_at(self.y, text_offset + 23, width=1, fg='magenta', attr='bold | italic')
            self.color_at(self.y, text_offset + 25, width=4, fg='red', attr='bold')
        else:
            self.addstr(self.y, text_offset, ' ' * 47)

    def destroy(self):
        super().destroy()
        self._daemon_running.clear()

    def print_width(self):
        return min(self.width, max((39 + len(process.host_info) for process in self.snapshots), default=79))

    def print(self):
        lines = ['', *self.header_lines()]

        if len(self.snapshots) > 0:
            key, reverse, *_ = self.ORDERS['natural']
            self.snapshots.sort(key=key, reverse=reverse)
            prev_device_index = None
            color = None
            for process in self.snapshots:
                device_index = process.device.index
                if prev_device_index != device_index:
                    color = process.device.snapshot.display_color
                    if prev_device_index is not None:
                        lines.append('├' + '─' * (self.width - 2) + '┤')
                    prev_device_index = device_index

                info = '{:>7} {} {:>7} {:>8} {:>3} {}'.format(
                    cut_string(process.pid, maxlen=7, padstr='.'), process.type,
                    cut_string(process.username, maxlen=7, padstr='+'),
                    process.gpu_memory_human, process.gpu_sm_utilization_string.replace('%', ''),
                    cut_string(process.host_info, padstr='..', maxlen=self.width - 39).ljust(self.width - 39)
                )
                is_zombie = (process.is_running and process.cmdline == ['Zombie Process'])
                no_permissions = (process.is_running and process.cmdline == ['No Permissions'])
                is_gone = (not process.is_running and process.cmdline == ['No Such Process'])
                if is_zombie or no_permissions or is_gone:
                    info = info.split(process.command)
                    if process.username != CURRENT_USER and not IS_SUPERUSER:
                        info = map(lambda item: colored(item, attrs=('dark',)), info)
                    info = colored(process.command, color=('red' if is_gone else 'yellow')).join(info)
                elif process.username != CURRENT_USER and not IS_SUPERUSER:
                    info = colored(info, attrs=('dark',))
                lines.append('│ {} {} │'.format(colored('{:>3}'.format(device_index), color), info))

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
        direction = event.wheel_direction()
        if event.shift():
            self.host_offset += 2 * direction
        else:
            self.selected.move(direction=direction)
        return True

    def __contains__(self, item):
        if self.parent.visible and isinstance(item, MouseEvent):
            return True
        return super().__contains__(item)
