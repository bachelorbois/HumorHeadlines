import numpy as np
from lib.parsing import Headline
from lib.features import Feature
from lib.features.embeddingContainer import EmbeddingContainer

class DistanceFeature(Feature):
    """A feature encoding the cosine distance between the edit and the edited word in embedding space.
    The embedding used can be set in the EmbeddingContainer static class.
    """
    @classmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        """Computes the distance feature for the provided Headline.

        Args:
            HL (Headline): The headline the distance should be computed for.

        Returns:
            np.ndarray: Distance encoded as a vector
        """
        EmbeddingContainer.init()

        w1 = HL.sentence[HL.word_index]
        w2 = HL.edit

        if w1 not in EmbeddingContainer.FT or w2 not in EmbeddingContainer.FT:
            return np.array([0])

        e1, e2 = EmbeddingContainer.lookup(w1, w2)

        cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

        return np.array([cos])
