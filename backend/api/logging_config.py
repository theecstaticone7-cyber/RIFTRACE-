"""Shared logging setup for the RiftRace API, so request/error logs are
consistent across routers, services, and main.py.
"""

import logging

logger = logging.getLogger("riftrace")


def configure_logging() -> None:
    if logger.handlers:
        return  # avoid duplicate handlers if called more than once (e.g. --reload)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
