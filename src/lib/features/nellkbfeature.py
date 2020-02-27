import numpy as np
from collections import defaultdict
from lib.parsing import Headline
from lib.features import Feature

class NellKbFeature(Feature):
    word2int = defaultdict(dict)
    cat2int = defaultdict(int)
    #construct vocab dictionary from Nell knowledge base.
    with open('../data/NELL/NELLVocab.txt', 'r') as f:
        vocab = f.readlines()
        for idx, e in enumerate(vocab):
            #extract the actual concept and its generalization (category).
            #and correct spaces.
            e = e.strip()
            e = e.replace('_', ' ')
            e = e.split(':') 
            word = e[-1]; category = e[-2]
            try:
                word2int[category][word] = idx
            except KeyError:
                print('fuck!')
        for idx, c in enumerate(word2int.keys()):
            try:
                cat2int[c] = idx+1
            except KeyError:
                print('category KeyError exception captured!')
    @classmethod
    def compute_feature(cls, HL: Headline) -> np.ndarray:
        entities = []
        generalizations = []
        sent = HL.sentence
        found = False
        sent[HL.word_index] = HL.edit
        #print(sent)
        for token in sent:
            found = False
            token = token.lower()
            for c in cls.word2int.keys():
                for k, v in cls.word2int[c].items():
                    if k == token:
                        entities.append(v)
                        generalizations.append(cls.cat2int[c])
                        found = True
                        break
                if found:
                    break
            if not found:
                entities.append(0)
                generalizations.append(0)
        a = np.array(generalizations)
        b = np.array(entities)
        stacked = np.stack((a,b))
        #print(a.shape, stacked.shape)
        return stacked
