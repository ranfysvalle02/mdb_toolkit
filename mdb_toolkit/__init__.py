# mdb_toolkit/__init__.py

import logging

logger = logging.getLogger(__name__)
logger.info("Initializing mdb_toolkit package")

from .core import CustomMongoClient, Node, Edge

__all__ = [
    'CustomMongoClient',
    'Node',
    'Edge'
]
