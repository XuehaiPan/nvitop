# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Caching utilities."""

from __future__ import annotations

import builtins
import functools
import time
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, NamedTuple, overload


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sized
    from collections.abc import Set as AbstractSet
    from typing import TypeVar
    from typing_extensions import (
        ParamSpec,  # Python 3.10+
        Self,  # Python 3.11+
    )

    _P = ParamSpec('_P')
    _T = TypeVar('_T')


__all__ = ['ttl_cache']


class _CacheInfo(NamedTuple):
    """A named tuple representing the cache statistics."""

    hits: int
    misses: int
    maxsize: int
    currsize: int


try:
    from functools import _make_key
except ImportError:

    class _HashedSeq(list):
        """This class guarantees that hash() will be called no more than once per element."""

        __slots__ = ('__hashvalue',)

        def __init__(
            self,
            seq: tuple[Any, ...],
            hash: Callable[[Any], int] = builtins.hash,  # pylint: disable=redefined-builtin
        ) -> None:
            """Initialize the hashed sequence."""
            self[:] = seq
            self.__hashvalue = hash(seq)

        def __hash__(self) -> int:  # type: ignore[override]
            """Return the hash value of the hashed sequence."""
            return self.__hashvalue

    _KWD_MARK = object()

    # pylint: disable-next=too-many-arguments
    def _make_key(  # type: ignore[misc]
        args: tuple[Hashable, ...],
        kwds: dict[str, Hashable],
        typed: bool,
        *,
        kwd_mark: tuple[object, ...] = (_KWD_MARK,),
        fasttypes: AbstractSet[type] = frozenset({int, str}),
        tuple: type[tuple] = builtins.tuple,  # pylint: disable=redefined-builtin
        type: type[type] = builtins.type,  # pylint: disable=redefined-builtin
        len: Callable[[Sized], int] = builtins.len,  # pylint: disable=redefined-builtin
    ) -> Hashable:
        """Make a cache key from optionally typed positional and keyword arguments."""
        key = args
        if kwds:
            key += kwd_mark
            for item in kwds.items():
                key += item
        if typed:
            key += tuple(type(v) for v in args)
            if kwds:
                key += tuple(type(v) for v in kwds.values())
        elif len(key) == 1 and type(key[0]) in fasttypes:
            return key[0]
        return _HashedSeq(key)


@dataclass
class _TTLCacheLink:  # pylint: disable=too-few-public-methods
    __slots__ = ('expires', 'key', 'next', 'prev', 'value')

    prev: Self
    next: Self  # pylint: disable=redefined-builtin
    key: Hashable
    value: Any
    expires: float


@overload
def ttl_cache(
    maxsize: int | None = 128,
    *,
    ttl: float = 600.0,
    timer: Callable[[], float] = time.monotonic,
    typed: bool = False,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@overload
def ttl_cache(
    maxsize: Callable[_P, _T],
    *,
    ttl: float = 600.0,
    timer: Callable[[], float] = time.monotonic,
    typed: bool = False,
) -> Callable[_P, _T]: ...


# pylint: disable-next=too-many-statements
def ttl_cache(
    maxsize: int | Callable[_P, _T] | None = 128,
    *,
    ttl: float = 600.0,
    timer: Callable[[], float] = time.monotonic,
    typed: bool = False,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]] | Callable[_P, _T]:
    """Time aware cache decorator."""
    if isinstance(maxsize, int):
        # Negative maxsize is treated as 0
        maxsize = max(0, maxsize)
    elif callable(maxsize) and isinstance(typed, bool):
        # The user_function was passed in directly via the maxsize argument
        func, maxsize = maxsize, 128
        return ttl_cache(maxsize, ttl=ttl, timer=timer, typed=typed)(func)
    elif maxsize is not None:
        raise TypeError('Expected first argument to be an integer, a callable, or None')

    if ttl < 0.0:
        raise ValueError('TTL must be a non-negative number')
    if not callable(timer):
        raise TypeError('Timer must be a callable')

    if maxsize == 0 or maxsize is None:
        return functools.lru_cache(maxsize=maxsize, typed=typed)  # type: ignore[return-value]

    # pylint: disable-next=too-many-statements,too-many-locals
    def wrapper(func: Callable[_P, _T], /) -> Callable[_P, _T]:
        cache: dict[Any, _TTLCacheLink] = {}
        cache_get = cache.get  # bound method to lookup a key or return None
        cache_len = cache.__len__  # get cache size without calling len()
        lock = RLock()  # because linked-list updates aren't thread-safe
        # root of the circular doubly linked list
        root = _TTLCacheLink(*((None,) * 5))  # type: ignore[arg-type]
        root.prev = root.next = root  # initialize by pointing to self
        hits = misses = 0
        full = False

        def unlink(link: _TTLCacheLink) -> _TTLCacheLink:
            with lock:
                link_prev, link_next = link.prev, link.next
                link_next.prev, link_prev.next = link_prev, link_next
            return link_next

        def append(link: _TTLCacheLink) -> _TTLCacheLink:
            with lock:
                last = root.prev
                last.next = root.prev = link
                link.prev, link.next = last, root
            return link

        def move_to_end(link: _TTLCacheLink) -> _TTLCacheLink:
            with lock:
                unlink(link)
                append(link)
            return link

        def expire() -> None:
            nonlocal full

            with lock:
                now = timer()
                front = root.next
                while front is not root and front.expires < now:
                    del cache[front.key]
                    front = unlink(front)
                full = cache_len() >= maxsize

        @functools.wraps(func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            # Size limited time aware caching
            nonlocal root, hits, misses, full

            key = _make_key(args, kwargs, typed)
            with lock:
                link = cache_get(key)
                if link is not None:
                    if timer() < link.expires:
                        hits += 1
                        return link.value
                    expire()

            misses += 1
            result = func(*args, **kwargs)
            expires = timer() + ttl
            with lock:
                if key in cache:
                    # Getting here means that this same key was added to the cache while the lock
                    # was released or the key was expired. Move the link to the front of the
                    # circular queue.
                    link = move_to_end(cache[key])
                    # We need only update the expiration time.
                    link.value = result
                    link.expires = expires
                else:
                    if full:
                        expire()
                    if full:
                        # Use the old root to store the new key and result.
                        root.key = key
                        root.value = result
                        root.expires = expires
                        # Empty the oldest link and make it the new root.
                        # Keep a reference to the old key and old result to prevent their ref counts
                        # from going to zero during the update. That will prevent potentially
                        # arbitrary object clean-up code (i.e. __del__) from running while we're
                        # still adjusting the links.
                        front = root.next
                        old_key = front.key
                        front.key = front.value = front.expires = None  # type: ignore[assignment]
                        # Now update the cache dictionary.
                        del cache[old_key]
                        # Save the potentially reentrant cache[key] assignment for last, after the
                        # root and links have been put in a consistent state.
                        cache[key], root = root, front
                    else:
                        # Put result in a new link at the front of the queue.
                        cache[key] = append(_TTLCacheLink(None, None, key, result, expires))  # type: ignore[arg-type]
                    full = cache_len() >= maxsize
            return result

        def cache_info() -> _CacheInfo:
            """Report cache statistics."""
            with lock:
                expire()
                return _CacheInfo(hits, misses, maxsize, cache_len())

        def cache_clear() -> None:
            """Clear the cache and cache statistics."""
            nonlocal hits, misses, full
            with lock:
                cache.clear()
                root.prev = root.next = root
                root.key = root.value = root.expires = None  # type: ignore[assignment]
                hits = misses = 0
                full = False

        wrapped.cache_info = cache_info  # type: ignore[attr-defined]
        wrapped.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapped.cache_parameters = lambda: {'maxsize': maxsize, 'typed': typed}  # type: ignore[attr-defined]
        return wrapped

    return wrapper
