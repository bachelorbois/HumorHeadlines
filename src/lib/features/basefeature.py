from abc import ABC, abstractmethod
import numpy as np
from lib.parsing import Headline

class Feature(ABC):
    """Abstract base feature class. Should be extended by all features.
    """
    @classmethod
    @abstractmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        """Computes the feature.
        Implemented by the concrete classes, this method is called by Headline when building feature vectors.

        Args:
            HL (Headline): A headline object, that the feature should be computed for

        Returns:
            np.ndarray: The computed feature vector
        """
        pass
