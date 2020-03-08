import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from lib.parsing import Headline
from lib.features import Feature
from lib.features.NELLWordnetVocab import generate_vocab_dict

class NellKbFeature(Feature):
    max_len = 20
    lemmatizer = WordNetLemmatizer()
    punct = set(string.punctuation)
    punct.add('—‘’$“”')
    punct = list(punct)
    punct.extend(['’s', "'s", "'m", "...", 'n’t', "`"])
    word2int, int2word = generate_vocab_dict("../data/NELL/NELLWordNetVocab_proc.txt")

    @classmethod
    def preprocess(cls, HL):
        sent = HL.sentence
        sent[HL.word_index] = HL.edit
        sent = [w.lower().replace("'s", "") for w in sent]
        try:
            sent = [cls.lemmatizer.lemmatize(w) for w in sent if w not in cls.punct]
        except Exception as e:
            print(e)
        return sent

    def pad(n, list_to_pad):
        if n >= 1:
            for _ in range(0, n):
                list_to_pad.append(0)

    @classmethod
    def compute_feature(cls, HL: Headline) -> np.ndarray:
        entities = []
        sentence = cls.preprocess(HL)
        # print(sentence)
        for token in sentence:
            try:
                entities.append(cls.word2int[token])
            except KeyError:
                pass
        
        curr_len = len(entities)
        
        if curr_len > cls.max_len:
            print(f"oh shit {curr_len}")
        
        pad_size = cls.max_len - curr_len
        cls.pad(pad_size, entities)
        
        a = np.array(entities)
        # print(a.shape)
        return a
