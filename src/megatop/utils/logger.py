import logging
import os

from megatop.utils.mpi import get_world

__all__ = [
    "logger",
]

try:
    # Get MPI information
    comm, rank, size = get_world()
except ImportError:
    rank = 0
    size = 1

log_format = (
    "%(asctime)s - Rank %(rank)d - %(levelname)s: %(message)s"
    if size > 1
    else "%(asctime)s - %(levelname)s: %(message)s"
)

logging.basicConfig(
    format=log_format,
    datefmt="%d-%b-%y %H:%M:%S",
)


# Custom filter to include rank in log messages
class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = rank
        return True


logger = logging.getLogger("megatop")
logger.addFilter(RankFilter())
logger.setLevel(os.getenv("LOGLEVEL", "INFO").upper())
