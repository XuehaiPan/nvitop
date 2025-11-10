# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from __future__ import annotations

import functools
import itertools
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, TypeVar

from nvitop.api import NA


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing_extensions import ParamSpec  # Python 3.11+

    _P = ParamSpec('_P')
    _T = TypeVar('_T')
    _S = TypeVar('_S')


__all__ = ['BufferedHistoryGraph', 'HistoryGraph']


BOUND_UPDATE_INTERVAL: float = 1.0

# fmt: off
VALUE2SYMBOL_UP: dict[tuple[int, int], str] = {
    (0, 0): ' ', (0, 1): '⢀', (0, 2): '⢠', (0, 3): '⢰', (0, 4): '⢸',
    (1, 0): '⡀', (1, 1): '⣀', (1, 2): '⣠', (1, 3): '⣰', (1, 4): '⣸',
    (2, 0): '⡄', (2, 1): '⣄', (2, 2): '⣤', (2, 3): '⣴', (2, 4): '⣼',
    (3, 0): '⡆', (3, 1): '⣆', (3, 2): '⣦', (3, 3): '⣶', (3, 4): '⣾',
    (4, 0): '⡇', (4, 1): '⣇', (4, 2): '⣧', (4, 3): '⣷', (4, 4): '⣿',
}
VALUE2SYMBOL_DOWN: dict[tuple[int, int], str] = {
    (0, 0): ' ', (0, 1): '⠈', (0, 2): '⠘', (0, 3): '⠸', (0, 4): '⢸',
    (1, 0): '⠁', (1, 1): '⠉', (1, 2): '⠙', (1, 3): '⠹', (1, 4): '⢹',
    (2, 0): '⠃', (2, 1): '⠋', (2, 2): '⠛', (2, 3): '⠻', (2, 4): '⢻',
    (3, 0): '⠇', (3, 1): '⠏', (3, 2): '⠟', (3, 3): '⠿', (3, 4): '⢿',
    (4, 0): '⡇', (4, 1): '⡏', (4, 2): '⡟', (4, 3): '⡿', (4, 4): '⣿',
}
# fmt: on
SYMBOL2VALUE_UP: dict[str, tuple[int, int]] = {v: k for k, v in VALUE2SYMBOL_UP.items()}
SYMBOL2VALUE_DOWN: dict[str, tuple[int, int]] = {v: k for k, v in VALUE2SYMBOL_DOWN.items()}
PAIR2SYMBOL_UP: dict[tuple[str, str], str] = {
    (s1, s2): VALUE2SYMBOL_UP[SYMBOL2VALUE_UP[s1][-1], SYMBOL2VALUE_UP[s2][0]]
    for s1, s2 in itertools.product(SYMBOL2VALUE_UP, repeat=2)
}
PAIR2SYMBOL_DOWN: dict[tuple[str, str], str] = {
    (s1, s2): VALUE2SYMBOL_DOWN[SYMBOL2VALUE_DOWN[s1][-1], SYMBOL2VALUE_DOWN[s2][0]]
    for s1, s2 in itertools.product(SYMBOL2VALUE_DOWN, repeat=2)
}
GRAPH_SYMBOLS: str = ''.join(
    sorted(set(itertools.chain(VALUE2SYMBOL_UP.values(), VALUE2SYMBOL_DOWN.values()))),
).replace(' ', '')


def grouped(
    iterable: Iterable[_T],
    size: int,
    fillvalue: _S = None,  # type: ignore[assignment]
) -> itertools.zip_longest[tuple[_T | _S, ...]]:
    it = iter(iterable)
    return itertools.zip_longest(*([it] * size), fillvalue=fillvalue)


class HistoryGraph:  # pylint: disable=too-many-instance-attributes
    MAX_WIDTH = 1024

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        upperbound: float,
        width: int,
        height: int,
        *,
        format: Callable[[float], str] = '{:.1f}'.format,  # pylint: disable=redefined-builtin
        max_format: Callable[[float], str] | None = None,
        baseline: float = 0.0,
        dynamic_bound: bool = False,
        min_bound: float | None = None,
        init_bound: float | None = None,
        upsidedown: bool = False,
    ) -> None:
        assert baseline < upperbound

        self.format: Callable[[float], str] = format
        if max_format is None:
            max_format = format
        self.max_format: Callable[[float], str] = max_format

        if dynamic_bound:
            if min_bound is None:
                min_bound = baseline + 0.1 * (upperbound - baseline)
            if init_bound is None:
                init_bound = upperbound
        else:
            assert min_bound is None
            assert init_bound is None
            min_bound = init_bound = upperbound
        self.baseline: float = baseline
        self.min_bound: float = min_bound
        self.max_bound: float = upperbound
        self.bound: float = init_bound
        self.next_bound_update_at: float = time.monotonic()
        self._width: int = width
        self._height: int = height

        self.maxlen: int = 2 * self.width + 1
        self.history: deque[float] = deque(
            [self.baseline - 0.1] * (2 * self.MAX_WIDTH + 1),
            maxlen=(2 * self.MAX_WIDTH + 1),
        )
        self.reversed_history: deque[float] = deque(
            [self.baseline - 0.1] * self.maxlen,
            maxlen=self.maxlen,
        )
        self._max_value_maintainer: deque[float] = deque(
            [self.baseline - 0.1] * self.maxlen,
            maxlen=self.maxlen,
        )
        self.last_retval: Any = None

        self.graph: list[str] = []
        self.last_graph: list[str] = []
        self.upsidedown: bool = upsidedown
        if upsidedown:
            self.value2symbol: dict[tuple[int, int], str] = VALUE2SYMBOL_DOWN
            self.pair2symbol: dict[tuple[str, str], str] = PAIR2SYMBOL_DOWN
        else:
            self.value2symbol = VALUE2SYMBOL_UP
            self.pair2symbol = PAIR2SYMBOL_UP

        self.write_lock = threading.Lock()
        self.remake_lock = threading.Lock()
        self.remake_graph()

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int) -> None:
        if self._width != value:
            assert isinstance(value, int)
            assert value >= 1
            self._width = value
            with self.write_lock:
                self.maxlen = 2 * self.width + 1
                self.reversed_history = deque(
                    (self.baseline - 0.1,) * self.maxlen,
                    maxlen=self.maxlen,
                )
                self._max_value_maintainer = deque(
                    (self.baseline - 0.1,) * self.maxlen,
                    maxlen=self.maxlen,
                )
                for history in itertools.islice(
                    self.history,
                    max(0, self.history.maxlen - self.maxlen),  # type: ignore[operator]
                    self.history.maxlen,
                ):
                    if self.reversed_history[-1] == self._max_value_maintainer[0]:
                        self._max_value_maintainer.popleft()
                    while (
                        len(self._max_value_maintainer) > 0
                        and self._max_value_maintainer[-1] < history
                    ):
                        self._max_value_maintainer.pop()
                    self.reversed_history.appendleft(history)
                    self._max_value_maintainer.append(history)
                self.remake_graph()

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        if self._height != value:
            assert isinstance(value, int)
            assert value >= 1
            self._height = value
            self.remake_graph()

    @property
    def graph_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @graph_size.setter
    def graph_size(self, value: tuple[int, int]) -> None:
        width, height = value
        assert isinstance(width, int)
        assert width >= 1
        assert isinstance(height, int)
        assert height >= 1
        self._height = height
        self._width = width - 1  # trigger force remake
        self.width = width

    @property
    def last_value(self) -> float:
        return self.reversed_history[0]

    @property
    def max_value(self) -> float:
        return self._max_value_maintainer[0]

    def last_value_string(self) -> str:
        last_value = self.last_value
        if last_value >= self.baseline:
            return self.format(last_value)
        try:
            return self.format(NA)  # type: ignore[arg-type]
        except ValueError:
            return NA

    __str__ = last_value_string

    def max_value_string(self) -> str:
        max_value = self.max_value
        if max_value >= self.baseline:
            return self.max_format(max_value)
        try:
            return self.max_format(NA)  # type: ignore[arg-type]
        except ValueError:
            return NA

    def add(self, value: float) -> None:
        if value is NA:  # type: ignore[comparison-overlap]
            value = self.baseline - 0.1
        assert isinstance(value, (int, float))

        with self.write_lock:
            if self.reversed_history[-1] == self._max_value_maintainer[0]:
                self._max_value_maintainer.popleft()
            while len(self._max_value_maintainer) > 0 and self._max_value_maintainer[-1] < value:
                self._max_value_maintainer.pop()
            self.reversed_history.appendleft(value)
            self._max_value_maintainer.append(value)
            self.history.append(value)

            new_bound = self.baseline + 1.25 * (self.max_value - self.baseline)
            new_bound = min(max(new_bound, self.min_bound), self.max_bound)
            timestamp = time.monotonic()
            if new_bound != self.bound and self.next_bound_update_at <= timestamp:
                self.bound = new_bound
                self.remake_graph()
                self.next_bound_update_at = timestamp + BOUND_UPDATE_INTERVAL
                return

            self.graph, self.last_graph = self.last_graph, self.graph
            bar_chart = self.make_bar_chart(self.reversed_history[1], value)
            for i, (line, char) in enumerate(zip(self.graph, bar_chart)):
                self.graph[i] = (line + char)[-self.width :]

    def remake_graph(self) -> None:
        with self.remake_lock:
            if self.max_value >= self.baseline:
                reversed_bar_charts = []
                for _, (value2, value1) in zip(
                    range(self.width),
                    grouped(self.reversed_history, size=2, fillvalue=self.baseline),
                ):
                    reversed_bar_charts.append(self.make_bar_chart(value1, value2))
                graph = list(map(''.join, zip(*reversed(reversed_bar_charts))))

                for i, line in enumerate(graph):
                    graph[i] = line.rjust(self.width)[-self.width :]

                self.graph = graph
                self.last_graph = list(map(self.shift_line, self.graph))
            else:
                self.graph = [' ' * self.width for _ in range(self.height)]
                self.last_graph = [' ' * (self.width - 1) for _ in range(self.height)]

    def make_bar_chart(self, value1: float, value2: float) -> list[str]:
        if self.bound <= self.baseline:
            return [' '] * self.height

        value1 = self.height * min((value1 - self.baseline) / (self.bound - self.baseline), 1.0)
        value2 = self.height * min((value2 - self.baseline) / (self.bound - self.baseline), 1.0)
        if value1 >= 0.0:
            value1 = max(value1, 0.2)
        if value2 >= 0.0:
            value2 = max(value2, 0.2)
        bar_charts = []
        for h in range(self.height):
            s1 = min(max(round(5 * (value1 - h)), 0), 4)
            s2 = min(max(round(5 * (value2 - h)), 0), 4)
            bar_charts.append(self.value2symbol[s1, s2])
        if not self.upsidedown:
            bar_charts.reverse()
        return bar_charts

    def shift_line(self, line: str) -> str:
        return ''.join(self.pair2symbol[p] for p in zip(line, line[1:]))

    def __getitem__(self, item: int) -> float:
        return self.reversed_history[item]

    def hook(
        self,
        func: Callable[_P, _T],
        *,
        get_value: Callable[[_T], float] | None = None,
    ) -> Callable[_P, _T]:
        @functools.wraps(func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            self.last_retval = retval = func(*args, **kwargs)
            value: float = get_value(retval) if get_value is not None else retval  # type: ignore[assignment]
            self.add(value)
            return retval

        wrapped.history = self  # type: ignore[attr-defined]
        return wrapped

    __call__ = hook


class BufferedHistoryGraph(HistoryGraph):
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        upperbound: float,
        width: int,
        height: int,
        *,
        format: Callable[[float], str] = '{:.1f}'.format,  # pylint: disable=redefined-builtin
        max_format: Callable[[float], str] | None = None,
        baseline: float = 0.0,
        dynamic_bound: bool = False,
        min_bound: float | None = None,
        init_bound: float | None = None,
        upsidedown: bool = False,
        interval: float = 1.0,
    ) -> None:
        assert interval > 0.0
        super().__init__(
            upperbound,
            width,
            height,
            format=format,
            max_format=max_format,
            baseline=baseline,
            dynamic_bound=dynamic_bound,
            min_bound=min_bound,
            init_bound=init_bound,
            upsidedown=upsidedown,
        )

        self.interval: float = interval
        self.start_time: float = time.monotonic()
        self.last_update_time: float = self.start_time
        self.buffer: list[float] = []

    @property
    def last_value(self) -> float:
        last_value = super().last_value
        if last_value < self.baseline and len(self.buffer) > 0:
            return sum(self.buffer) / len(self.buffer)
        return last_value

    def add(self, value: float) -> None:
        if value is NA:  # type: ignore[comparison-overlap]
            value = self.baseline - 0.1
        assert isinstance(value, (int, float))

        timestamp = time.monotonic()
        timedelta = timestamp - self.last_update_time
        if len(self.buffer) > 0 and timedelta >= self.interval:
            new_value = sum(self.buffer) / len(self.buffer)
            self.buffer.clear()
            last_value = self.reversed_history[0]
            if last_value >= self.baseline:
                n_interval = int(timedelta / self.interval)
                for i in range(1, n_interval):
                    super().add(last_value + (i / n_interval) * (new_value - last_value))
            super().add(new_value)

            self.last_update_time += (timedelta // self.interval) * self.interval
        self.buffer.append(value)
