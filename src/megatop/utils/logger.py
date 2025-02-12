import logging
import os

__all__ = [
    "logger",
]


logging.basicConfig(
    format="%(asctime)s - %(module)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("megatop")
logger.setLevel(os.getenv("LOGLEVEL", "INFO").upper())
