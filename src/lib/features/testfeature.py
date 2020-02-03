import numpy as np
from lib.features import Feature
from lib.parsing import Headline

class Testfeat(Feature):

    @classmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        return np.array(
            [0, 1, 2, 3, 4, 5, 4 ,3, 2, 1]
        )
