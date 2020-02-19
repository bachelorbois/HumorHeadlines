import os
import wget
import linecache
from zipfile import ZipFile
import numpy as np
import pickle
from lib.parsing import Headline
from lib.features import Feature
from lib.features.embeddingContainer import EmbeddingContainer

class DistanceFeature(Feature):
    @classmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        EmbeddingContainer.init()

        w1 = HL.sentence[HL.word_index]
        w2 = HL.edit

        if w1 not in EmbeddingContainer.FT or w2 not in EmbeddingContainer.FT:
            return np.array([0])

        e1, e2 = EmbeddingContainer.lookup(w1, w2)

        cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

        return np.array([cos])
