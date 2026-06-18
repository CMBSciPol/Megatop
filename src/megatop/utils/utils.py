import os
import tracemalloc

import psutil

from .logger import logger


def MemoryUsage(message: str = "") -> None:
    """'
    Prints the memory usage of the current process.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments from the command line.
        In particular looks for the verbose flag. If false, the function does nothing.
    message : str, optional
        The message to print. The default is ''.

    Returns
    -------
    None

    """
    current, peak = tracemalloc.get_traced_memory()
    message_all = (
        message + f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB"
    )
    logger.info(message_all)


def PSMemoryUsage(message=""):
    process = psutil.Process(os.getpid())

    rss = process.memory_info().rss / 1024**2  # MB

    logger.info(f"{message} RSS memory: {rss:.1f} MB")
