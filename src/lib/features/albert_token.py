import numpy as np
import tensorflow_hub as hub
from lib.parsing import Headline
from lib.features import Feature
from lib.tokenizer import FullSentencePieceTokenizer
from lib.features.tokenizerContainer import TokenizerContainer

class AlbertTokenizer(Feature):
    MAX_LEN = 128

    @classmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        """Computes the feature.
        Implemented by the concrete classes, this method is called by Headline when building feature vectors.

        Args:
            HL (Headline): A headline object, that the feature should be computed for

        Returns:
            np.ndarray: The computed feature vector
        """
        TokenizerContainer.init()
        tokens, input_ids = cls.get_ids(HL.GetSentWithoutEdit())
        segments = cls.get_segments(tokens)
        masks = cls.get_masks(tokens)
        input_ids.extend(segments)
        input_ids.extend(masks)
        return np.array(input_ids)

    @classmethod
    def get_masks(cls, tokens):
        """Mask for padding"""
        if len(tokens)>cls.MAX_LEN:
            raise IndexError("Token length more than max seq length!")
        return [1]*len(tokens) + [0] * (cls.MAX_LEN - len(tokens))

    @classmethod
    def get_segments(cls, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens)>cls.MAX_LEN:
            raise IndexError(f"Token length {len(tokens)} more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (cls.MAX_LEN - len(tokens))

    @classmethod
    def get_ids(cls, text):
        """Token ids from Tokenizer vocab"""
        tokens = TokenizerContainer.TOKENIZER.tokenize(text)
        token_ids = TokenizerContainer.TOKENIZER.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (cls.MAX_LEN-len(token_ids))
        return tokens, input_ids
        