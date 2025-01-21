from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from typing import ParamSpec, TypeVar

import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    pass


Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")


def requires_mpi4py(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    if find_spec("mpi4py") is not None:
        return func

    @wraps(func)
    def deferred_func(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        msg = "Missing optional library 'mpi4py', part of the 'mpi' dependency group."
        raise ImportError(msg)

    return deferred_func


@requires_mpi4py
def MPISUM(array, comm, rank, root):
    """
    Reduces an array using the SUM operator to the root process using MPI.

    Parameters
    ----------
    array : np.ndarray
        The array to reduce.
    comm : MPI.COMM_WORLD
        The MPI communicator.
    rank : int
        The rank of the process.
    root : int
        The root process.

    Returns
    -------
    array_recvbuf : np.ndarray
        The reduced array.

    """

    if rank == root:
        array_recvbuf = np.zeros_like(array)
    else:
        array_recvbuf = None

    array = np.ascontiguousarray(array)

    comm.Reduce(array, array_recvbuf, op=MPI.SUM, root=root)

    return array_recvbuf


@requires_mpi4py
def MPIGATHER(array, comm, rank, size, root):
    """
    Gathers an array to the root process using MPI.

    Parameters
    ----------
    array : np.ndarray
        The array to gather.
    comm : MPI.COMM_WORLD
        The MPI communicator.
    rank : int
        The rank of the process.
    size : int
        The size of the communicator.
    root : int
        The root process.

    Returns
    -------
    array_recvbuf : np.ndarray
        The gathered array.

    """

    array = np.ascontiguousarray(array)

    array_recvbuf = None
    if rank == 0:
        shape_recvbuf_array = (size, *array.shape)
        array_recvbuf = np.empty(shape_recvbuf_array)

        # Ensure recvbuf is contiguous
        # to make sure the comm.Gather() works correctly
        array_recvbuf = np.ascontiguousarray(array_recvbuf)

    comm.Gather(array, array_recvbuf, root=root)

    return array_recvbuf
