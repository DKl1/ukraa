import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseEncoder:
 
    def __init__(self):
        pass

    def encode(self, sentences: List[str], lang: Optional[str] = None):

        raise NotImplementedError("encode() must be implemented in subclasses")

    def __repr__(self):
        return f"{self.__class__.__name__}()"
