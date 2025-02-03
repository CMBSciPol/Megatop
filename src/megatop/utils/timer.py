import functools
import time
from queue import LifoQueue
from threading import Lock
from typing import ClassVar

from .logger import logger

__all__ = [
    "Timer",
    "function_timer",
]


class _SingletonReInit(type):
    """Thread-safe singleton implementation with reinitialization."""

    _instances: ClassVar = {}
    _lock: ClassVar[Lock] = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            else:
                instance = cls._instances[cls]
                # allow reinitialization, useful if using Timer as a context manager
                instance.__init__(*args, **kwargs)
        return instance


class Timer(metaclass=_SingletonReInit):
    """Timer class to measure elapsed time."""

    __slots__ = [
        "__dict__",
        "_context_latest_thread",
        "_context_manager_threads",
        "_thread_starts",
    ]

    _lock_init: bool = False

    def __init__(self, thread: str | None = None):
        if not self._lock_init:
            # initialize dict and queue only once
            self._thread_starts: dict[str, int] = {}
            self._context_threads: LifoQueue[str] = LifoQueue()
            self._lock_init = True
        self._context_latest_thread = self._normalize_thread(thread)

    def __enter__(self) -> "Timer":
        self._context_threads.put(self._context_latest_thread)
        self.start(self._context_latest_thread)
        return self

    def __exit__(self, *_exc_info) -> None:
        last_context_manager_thread = self._context_threads.get()
        self.stop(last_context_manager_thread)

    def start(self, thread: str | None = None) -> None:
        """Start the Timer. Should always be followed by a stop() call later in the code."""
        thread = self._normalize_thread(thread)
        if thread in self._thread_starts:
            msg = f"Timer {thread} already exists."
            raise ValueError(msg)

        self._thread_starts[thread] = time.perf_counter_ns()

    def stop(self, thread: str | None = None) -> None:
        """Stop the Timer and log the time elapsed since the start() call."""
        thread = self._normalize_thread(thread)
        if thread not in self._thread_starts:
            msg = f"Timer {thread} does not exist."
            raise ValueError(msg)

        start_time_ns = self._thread_starts.pop(thread)
        elapsed_ns = time.perf_counter_ns() - start_time_ns
        msg = f"Elapsed time [{thread}]: {self._format_nanoseconds(elapsed_ns)}"
        logger.info(msg)

    @classmethod
    def _format_nanoseconds(cls, ns: int) -> str:
        # when possible, bypass divmod for performance
        us, ns = divmod(ns, 1000) if ns > 1000 else (0, ns)
        ms, us = divmod(us, 1000) if us > 1000 else (0, us)
        ss, ms = divmod(ms, 1000) if ms > 1000 else (0, ms)
        mm, ss = divmod(ss, 60) if ss > 60 else (0, ss)
        hh, mm = divmod(mm, 60) if mm > 60 else (0, mm)
        if hh > 0:
            # 1 h 02 m 03 s
            return f"{hh} h {mm:02} m {ss:02} s"
        if mm > 0:
            # 2 m 03 s
            return f"{mm} m {ss:02} s"
        if ss > 0:
            # 3.045 s
            return f"{ss}.{ms:03} s"
        if ms > 0:
            # 45.006 ms
            return f"{ms}.{us:03} ms"
        if us > 0:
            # 6.078 us
            return f"{us}.{ns:03} us"
        # sub-microsecond, just print nanoseconds
        return f"{ns} ns"

    def _normalize_thread(self, thread: str | None) -> str:
        return thread or "default"


def _get_function_with_arguments_as_thread_name(func, args, kwargs) -> str:
    thread_name = func.__name__
    if args or kwargs:
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        mapped_args = [f"{name}={value!r}" for name, value in zip(arg_names, args, strict=False)]
        mapped_kwargs = [f"{key}={value!r}" for key, value in kwargs.items()]
        all_mapped_args = ", ".join(mapped_args + mapped_kwargs)
        thread_name += f"({all_mapped_args})"
    return thread_name


def function_timer(thread: str | None = None):
    """Function decorator to measure the performance of a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread_name = thread or _get_function_with_arguments_as_thread_name(func, args, kwargs)
            with Timer(thread=thread_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
