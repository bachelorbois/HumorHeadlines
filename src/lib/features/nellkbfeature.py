import string
import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from bidict import bidict
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from lib.parsing import Headline
from lib.features import Feature

class NellKbFeature(Feature):
    word2int = defaultdict(dict)
    cat2int = defaultdict(int)
    lemmatizer = WordNetLemmatizer()
    punct = set(string.punctuation)
    punct.add('—‘’$“”')
    punct = list(punct)
    punct.extend(['’s', "'s", "'m", "...", 'n’t', "`"])
    #construct vocab dictionary from Nell knowledge base.
    with open('../data/NELL/NELLVocab.txt', 'r') as f:
        vocab = f.readlines()
        for idx, e in enumerate(vocab):
            #extract the actual concept and its generalization (category).
            #and correct spaces.
            e = e.strip()
            e = e.replace('_', ' ')
            e = e.split(':')
            word = e[-1]
            category = e[-2]
            try:
                word2int[category][word] = idx+1
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
        try:
            sent = [cls.lemmatizer.lemmatize(w) for w in sent if w not in cls.punct]
        except Exception as e:
            print(e)
        print(sent)
        for token in sent:
            found = False
            token = token.lower()
            for c in cls.word2int.keys():
                if found:
                    break
                else:
                    try:
                        entities.append(cls.word2int[c][token])
                        generalizations.append(cls.cat2int[c])
                        found = True
                    except KeyError:
                        continue
            if not found:
                entities.append(0)
                generalizations.append(0)
        a = np.array(generalizations)
        b = np.array(entities)
        stacked = np.stack((a,b))
        #print(a.shape, stacked.shape)
        return stacked
