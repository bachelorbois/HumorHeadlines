import string
import numpy as np
from lib.parsing import Headline
from lib.features import Feature

class SentLenFeature(Feature):
    """A feature encoding the length of the sentence.
    """
    #construct a stop-word list consists of punctuation characters .
    stpwds = string.punctuation
    #expand the string with additional symbols found occuring in sentences.
    stpwds += '—‘’$“”'
    stpwds = list(stpwds)
    #possessive nouns are separated into 2 tokens, which has an unintended effect on length.
    stpwds.extend(['’s', "'s", "'m", "...", 'n’t'])
    #limit defined in the original paper.
    max_len = 20

    @classmethod
    def compute_feature(cls, HL: Headline) -> np.ndarray:
        s = HL.sentence
        print(" ".join(s))
        #count for each word not specified in the stop-list
        len_wo_stpwds = sum([1 for w in s if w not in cls.stpwds])
        #len_stpwds = sum([1 for w in s if w in cls.stpwds])
        #check if max length conforms to the original paper standard
        if len_wo_stpwds > 20:
            print(f"fuck! {len_wo_stpwds} {s}")
        #scale the length such that it exists in the interval: ]0,1]
        scaled_len = len_wo_stpwds/cls.max_len
        return np.array([scaled_len])



