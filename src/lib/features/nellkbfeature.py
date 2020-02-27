import numpy as np
from collections import defaultdict
from lib.parsing import Headline
from lib.features import Feature

class NellKbFeature(Feature):
    word2int = defaultdict()
    int2word = defaultdict()
    word2int['UNK'] = 0
    int2word[0] = ['UNK']
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
            word2int[word] = idx+1
            int2word[idx+1] = word
    @classmethod
    def compute_feature(cls, HL: Headline) -> np.ndarray:
        result = []
        sent = HL.sentence
        sent[HL.word_index] = HL.edit
        for token in sent:
            token = token.lower()
            if token in cls.word2int.keys():
                result.append(cls.word2int[token])
                #print(token)
            #if ' ' in token:
                #print(token)
            else:
               result.append(0)
        return np.array(result)
