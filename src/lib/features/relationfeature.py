import numpy as np
from collections import defaultdict
from lib.features import Feature
from lib.parsing import Headline

class RelationFeature(Feature):
    word2int = defaultdict(int)
    int2word = defaultdict(str)
    word2int['UNK'] = 0
    int2word[0] = 'UNK'
    with open('../data/NELL/NELLVocab.txt', 'r') as f:
        vocab = f.readlines()
        for idx, e in enumerate(vocab):
            e = e.strip()
            e = e.replace('_', ' ')
            e = e.split(':')
            word = e[-1]
            try:
                word2int[word] = idx+1
                int2word[idx+1] = word
            except KeyError:
                print('fuck!')
    data = np.load("../data/NELL/preproc/triples.npy") 
    labels = np.load("../data/NELL/preproc/labels.npy")
    boo_labels = (labels == 1)
    data = data[boo_labels]
    triplet_dict = {(s, o):p for s, p, o in data}
    
    @classmethod
    def compute_feature(cls, HL: Headline) -> np.ndarray:
        replaced = HL.sentence[HL.word_index]
        replacement = HL.edit
        replaced_id = cls.word2int[replaced]
        replacement_id = cls.word2int[replacement]
        result = []
        if replaced_id !=0 and replacement_id != 0:
            try:
                result.append(cls.triplet_dict[(replaced_id, replacement_id)])
            except KeyError:
                print(f'Relation for ({replaced}, {replacement}) not found!')
            try:
                result.append(cls.triplet_dict[(replacement_id, replaced_id)])
            except KeyError:
                print(f'Relation for ({replacement}, {replaced}) not found!')
        else:
            print(f"Entity not found!")
        return np.array(result)
