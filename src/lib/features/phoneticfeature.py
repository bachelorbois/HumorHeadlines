import numpy as np
from Levenshtein import StringMatcher
from g2p_en import G2p
from lib.features import Feature
from lib.parsing import Headline

class PhoneticFeature(Feature):
    # initialize grapheme to phoneme
    g2p = G2p()
    # error counter
    #counter = 0

    @classmethod
    def compute_feature(cls, HL: Headline) -> np.ndarray:
        # replaced word & replacement word.
        words = [HL.sentence[HL.word_index], HL.edit]
        # transcibe each token to arpabet.
        phones = [" ".join(cls.g2p(w.lower())) for w in words]
#         for i, w in enumerate(words):
            # try:
                # s = " "
                # words[i] = s.join(cls.g2p(w))

            # except KeyError:
                # # print erroneous key
                # print(w)
                # # tracks and prints errors
                # cls.counter += 1
                # print(cls.counter)
        # calculate levenshtein distance between the two pronunciation.
        levenshtein_dist = StringMatcher.distance(*phones)
        # scale using the max difference in "word length"
        scale_factor = max([len(w) for w in phones])
        scaled_dist = levenshtein_dist/scale_factor
        return np.array([scaled_dist])
