# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name

import curses
import getpass
import signal
import threading
import time
from collections import OrderedDict, namedtuple
from operator import attrgetter, xor

from cachetools.func import ttl_cache

from ...core import NA, host, GpuProcess, Snapshot
from ..lib import Displayable
from ..utils import colored, cut_string


CURRENT_USER = getpass.getuser()
if host.WINDOWS:  # pylint: disable=no-member
    import ctypes
    IS_SUPERUSER = bool(ctypes.windll.shell32.IsUserAnAdmin())
else:
    import os
    IS_SUPERUSER = ((os.geteuid() == 0) if hasattr(os, 'geteuid') else False)


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
                if direction > 0:
                    self.index = 0
                else:
                    self.index = len(processes) - 1
            else:
                self.index = min(max(0, self.index + direction), len(processes) - 1)
            self.process = processes[self.index]
        else:
            self.clear()

    def owned(self):
        return self.is_set() and (IS_SUPERUSER or self.process.username() == CURRENT_USER)

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
        if self.index is not None and self.process is not None:
            return True
        self.clear()
        return False

    __bool__ = is_set

    def is_same(self, process):
        try:
            return self.identity == process.identity
        except AttributeError:
            try:
                return self.identity == process._ident  # pylint: disable=protected-access
            except AttributeError:
                pass
        except TypeError:
            pass

        return False

    __eq__ = is_same

    def is_same_on_host(self, process):
        try:
            return self.identity[:2] == process.identity[:2]
        except AttributeError:
            try:
                return self.identity[:2] == process._ident[:2]  # pylint: disable=protected-access
            except AttributeError:
                pass
        except TypeError:
            pass

        return False


Order = namedtuple('Order', ['key', 'reverse', 'offset', 'column', 'previous', 'next'])


class ProcessPanel(Displayable):
    SNAPSHOT_INTERVAL = 0.7
    ORDERS = {
        'natural': Order(
            key=attrgetter('device.index', '_gone', 'username', 'pid'),
            reverse=False, offset=3, column='ID', previous='time', next='pid'
        ),
        'pid': Order(
            key=attrgetter('_gone', 'pid', 'device.index'),
            reverse=False, offset=9, column='PID', previous='natural', next='username'
        ),
        'username': Order(
            key=attrgetter('_gone', 'username', 'pid', 'device.index'),
            reverse=False, offset=18, column='USER', previous='pid', next='gpu_memory'
        ),
        'gpu_memory': Order(
            key=attrgetter('_gone', 'gpu_memory', 'gpu_sm_utilization', 'cpu_percent', 'pid', 'device.index'),
            reverse=True, offset=24, column='GPU-MEM', previous='username', next='sm_utilization'
        ),
        'sm_utilization': Order(
            key=attrgetter('_gone', 'gpu_sm_utilization', 'gpu_memory', 'cpu_percent', 'pid', 'device.index'),
            reverse=True, offset=33, column='SM', previous='gpu_memory', next='cpu_percent'
        ),
        'cpu_percent': Order(
            key=attrgetter('_gone', 'cpu_percent', 'memory_percent', 'pid', 'device.index'),
            reverse=True, offset=37, column='%CPU', previous='sm_utilization', next='memory_percent'
        ),
        'memory_percent': Order(
            key=attrgetter('_gone', 'memory_percent', 'cpu_percent', 'pid', 'device.index'),
            reverse=True, offset=43, column='%MEM', previous='cpu_percent', next='time'
        ),
        'time': Order(
            key=attrgetter('_gone', 'running_time', 'pid', 'device.index'),
            reverse=True, offset=49, column='TIME', previous='memory_percent', next='natural'
        ),
    }

    def __init__(self, devices, compact, win=None, root=None):
        super().__init__(win, root)

        self.devices = devices
        GpuProcess.CLIENT_MODE = True

        self._compact = compact
        self.width = max(79, root.width)
        self.height = self._full_height = self.compact_height = 7

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
        self._daemon_started = threading.Event()

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
        if self.compact != value or self._compact != value:
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
            self.host_offset = max(-1, min(self.host_offset, info_length - self.width + 38))
            if old_host_offset not in (self.host_offset, 1024):
                curses.beep()

            if self.selected.is_set():
                identity = self.selected.identity
                self.selected.clear()
                for i, process in enumerate(snapshots):
                    if process.identity == identity:
                        self.selected.index = i
                        self.selected.process = process
                        break

    @ttl_cache(ttl=2.0)
    def take_snapshots(self):
        GpuProcess.clear_host_snapshots()
        snapshots = list(filter(None, map(GpuProcess.as_snapshot, self.processes.values())))

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
        self._daemon_started.wait()
        while self._daemon_started.is_set():
            self.take_snapshots()
            time.sleep(self.SNAPSHOT_INTERVAL)

    def header_lines(self):
        header = [
            '╒' + '═' * (self.width - 2) + '╕',
            '│ {} │'.format('Processes:'.ljust(self.width - 4)),
            '│ GPU    PID      USER  GPU-MEM %SM  {} │'.format('  '.join(self.host_headers).ljust(self.width - 39)),
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
        if not self._daemon_started.is_set():
            self._daemon_started.set()
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

        self.addstr(self.y + 3, self.x + 1, ' GPU    PID      USER  GPU-MEM %SM  ')
        host_offset = max(self.host_offset, 0)
        command_offset = max(14 + len(self.host_headers[-2]) - host_offset, 0)
        if command_offset > 0:
            host_headers = '  '.join(self.host_headers)
            self.addstr(self.y + 3, self.x + 37, '{}'.format(host_headers[host_offset:].ljust(self.width - 39)))
        else:
            self.addstr(self.y + 3, self.x + 37, '{}'.format('COMMAND'.ljust(self.width - 39)))

        _, reverse, offset, column, *_ = self.ORDERS[self.order]
        column_width = len(column)
        reverse = xor(reverse, self.reverse)
        indicator = '▼' if reverse else '▲'
        if (self.order == 'natural' and reverse) or self.order in ('pid', 'username', 'gpu_memory'):
            self.addstr(self.y + 3, self.x + offset - 1, column + indicator)
            self.color_at(self.y + 3, self.x + offset - 1, width=column_width, attr='bold | underline')
            self.color_at(self.y + 3, self.x + offset + column_width - 1, width=1, attr='bold')
        elif self.order != 'natural':
            offset -= host_offset
            if self.order == 'time':
                offset += len(self.host_headers[-2]) - 4
            if offset > 37 or host_offset == 0:
                self.addstr(self.y + 3, self.x + offset - 1, column + indicator)
                self.color_at(self.y + 3, self.x + offset - 1, width=column_width, attr='bold | underline')
            elif offset <= 37 < offset + column_width:
                self.addstr(self.y + 3, self.x + 37, (column + indicator)[38 - offset:])
                if offset + column_width >= 39:
                    self.color_at(self.y + 3, self.x + 37, width=offset + column_width - 38, attr='bold | underline')
            if offset + column_width >= 38:
                self.color_at(self.y + 3, self.x + offset + column_width - 1, width=1, attr='bold')
        else:
            self.color_at(self.y + 3, self.x + 2, width=3, attr='bold')

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
                    host_info = cut_string(host_info, padstr='..', maxlen=self.width - 38)
                else:
                    host_info = host_info[self.host_offset:self.host_offset + self.width - 38]
                self.addstr(y, self.x,
                            '│ {:>3} {:>6} {} {:>7} {:>8} {:>3} {} │'.format(
                                device_index, cut_string(process.pid, maxlen=6, padstr='.'),
                                process.type, cut_string(process.username, maxlen=7, padstr='+'),
                                process.gpu_memory_human, process.gpu_sm_utilization_string.replace('%', ''),
                                host_info.ljust(self.width - 38)
                            ))
                if self.host_offset > 0:
                    self.addstr(y, self.x + 36, ' ')

                is_zombie = (process.is_running and process.cmdline == ['Zombie Process'])
                is_gone = (not process.is_running and process.cmdline == ['No Such Process'])
                if (is_zombie or is_gone) and command_offset == 0:
                    self.addstr(y, self.x + 37, process.command)

                if self.selected.is_same(process):
                    self.color_at(y, self.x + 1, width=self.width - 2, fg='cyan', attr='bold | reverse')
                    self.selected.within_window = (0 <= y < self.root.termsize[0])
                else:
                    if self.selected.is_same_on_host(process):
                        self.addstr(y, self.x + 1, '=')
                        self.color_at(y, self.x + 1, width=1, attr='bold | blink')
                    self.color_at(y, self.x + 2, width=3, fg=color)
                    if process.username != CURRENT_USER and not IS_SUPERUSER:
                        self.color_at(y, self.x + 5, width=self.width - 6, attr='dim')
                    if is_zombie or is_gone:
                        self.color_at(y, self.x + 37 + command_offset, width=15, fg=('red' if is_gone else 'yellow'))
                y += 1
            self.addstr(y, self.x, '╘' + '═' * (self.width - 2) + '╛',)
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

    def finalize(self):
        self.need_redraw = False

    def destroy(self):
        super().destroy()
        self._daemon_started.clear()

    def print_width(self):
        return min(self.width, max((38 + len(process.host_info) for process in self.snapshots), default=79))

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

                info = '{:>6} {} {:>7} {:>8} {:>3} {}'.format(
                    cut_string(process.pid, maxlen=6, padstr='.'), process.type,
                    cut_string(process.username, maxlen=7, padstr='+'),
                    process.gpu_memory_human, process.gpu_sm_utilization_string.replace('%', ''),
                    cut_string(process.host_info, padstr='..', maxlen=self.width - 38).ljust(self.width - 38)
                )
                is_zombie = (process.is_running and process.cmdline == ['Zombie Process'])
                is_gone = (not process.is_running and process.cmdline == ['No Such Process'])
                if is_zombie or is_gone:
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
            self.host_offset += direction
        else:
            self.selected.move(direction=direction)
        return True
