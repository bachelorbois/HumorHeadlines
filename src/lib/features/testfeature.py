import numpy as np
from lib.features import Feature
from lib.parsing import Headline
from random import randint

class TestFeature(Feature):
    FROM = 1
    TO = 6

    @classmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        return np.array(
            [randint(cls.FROM, cls.TO), 1, 2, 3, 4, 5, 4 ,3, 2, 1]
        )
