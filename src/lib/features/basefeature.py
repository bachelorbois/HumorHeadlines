from abc import ABC, abstractmethod
import numpy as np
from lib.parsing import Headline

class Feature(ABC):

    @classmethod
    @abstractmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        pass
