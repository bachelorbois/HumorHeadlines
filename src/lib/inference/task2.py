from tensorflow.keras import models
import tensorflow_hub as hub
import numpy as np
from lib.features.sentlenfeature import SentLenFeature
from lib.features.wordclusterfeature import ClusterFeature
from lib.features.distancefeature import DistanceFeature
from lib.features.positionfeature import PositionFeature
from lib.features.phoneticfeature import PhoneticFeature
from lib.parsing.parser import read_task2_pb, CandidateCollection
from lib.models.module.functionmodule import sigmoid_3
import pickle
import os


class Task2Inference:
    EMBED_FILE = "../data/embeddings/wiki-news-300d-1M"
    def __init__(self, model_path : str, test_data : str):
        self.humor = models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer, 'sigmoid_3': sigmoid_3})
        print("Loading candidate collection...")
        self.data = self.load_data(test_data)
        print("Loading fastText Embedings...")
        self.fastTextEmbeds = self.load_embeddings()

        features = [PhoneticFeature, PositionFeature, DistanceFeature, ClusterFeature, SentLenFeature]

        self.data.AddFeatures(features)


    def predict(self, path):
        features = self.data.GetFeatureVectors()
        HL1Features = features[:, 0]
        HL2Features = features[:, 1]

        token = self.data.GetTokenizedWEdit()
        HL1Tokens = self.process_sentences([row[0] for row in token], self.fastTextEmbeds)
        HL2Tokens = self.process_sentences([row[0] for row in token], self.fastTextEmbeds)

        sentences = self.data.GetEditedSentences()
        HL1Sentences = sentences[:, 0]
        HL2Sentences = sentences[:, 1]

        print("Predicting on HL1...")
        HL1Preds = self.humor.predict({"feature_input": HL1Features, "token_input": HL1Tokens, "string_input": HL1Sentences}).flatten()
        print("Predicting on HL2...")
        HL2Preds = self.humor.predict({"feature_input": HL2Features, "token_input": HL2Tokens, "string_input": HL2Sentences}).flatten()


        labels = (HL1Preds < HL2Preds).astype(int)
        labels += 1

        print(labels[:10])

        ids = self.data.GetIDS().astype(str)
        print(ids[:10], ids.dtype)
        out = np.stack((ids, labels), axis=-1)

        print(f'Saving to file {path}...')
        with open(path, 'w') as f:
            f.write('id,pred\n')
            for line in out:
                f.write(f'{line[0]},{line[1]}\n')
    
    @staticmethod
    def process_sentences(tokenized_sentences, dictionary):
        proc_sentences = [] 
        for sentence in tokenized_sentences:
            agg = np.zeros((300))
            for token in sentence:
                try:
                    agg = np.add(agg, dictionary[token])
                except KeyError:
                    agg = np.add(agg, dictionary['UNK'])
            proc_sentences.append(agg/len(sentence))

        return np.array(proc_sentences)

    
    def load_embeddings(self):
        dictionary = {}
        if not os.path.isfile(self.EMBED_FILE + '.p'):
            with open(self.EMBED_FILE + '.vec', 'r') as infile:
                for line in infile:
                    line = line.split()
                    word = line[0]
                    emb = np.array(line[1:], dtype='float')
                    dictionary[word] = emb

            dictionary['UNK'] = np.array(list(dictionary.values())).mean()

            with open(self.EMBED_FILE + '.p', 'wb') as f:
                pickle.dump(dictionary, f)
        else:
            with open(self.EMBED_FILE + '.p', 'rb') as f:
                dictionary = pickle.load(f)
        
        return dictionary


    @staticmethod
    def load_data(path) -> CandidateCollection:
        with open(path, 'rb') as fd:
            data = read_task2_pb(fd)

        return data