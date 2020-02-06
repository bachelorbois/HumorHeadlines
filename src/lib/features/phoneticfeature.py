from lib.features import Feature
from lib.parsing import Headline
import nltk
from Levenshtein import StringMatcher
import numpy as np

class PhoneticFeature(Feature):
    arpabet = nltk.corpus.cmudict.dict()
    scale_factor = 100 
    @classmethod
    def compute_feature(cls, HL : Headline):
        words = [HL.sentence[HL.word_index], HL.edit]
        #words = [" ".join(sum(cls.arpabet[w.lower()], [])) for w in words]
        for i, w in enumerate(words):
            try:
                words[i] = " ".join(sum(cls.arpabet[w.lower()], []))
            except KeyError as e:
                print(e)
                break

        return np.array(
                [StringMatcher.distance(*words)/cls.scale_factor]
                )

        


