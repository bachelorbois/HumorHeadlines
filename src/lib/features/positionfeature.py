import numpy as np
from lib.features import Feature
from lib.parsing import Headline

class PositionFeature(Feature):
    """A feature encoding the relative position of the edit in the sentence.
    """
    SCALING_FACTOR = 1

    @classmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        return np.array(
            [HL.word_index / len(HL) * cls.SCALING_FACTOR]
        )
