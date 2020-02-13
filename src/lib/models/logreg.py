import lib
import os
from lib.parsing import Headline
from lib.features import Feature
import numpy as np
import linecache
from tensorflow.keras import layers, Sequential, metrics
from tensorflow.keras.layers import Dense
from lib.models.module.functionmodule import sigmoid_3
import csv
from sklearn.model_selection import train_test_split



class LogReg():
    FT = None
    DATA_FILE = "../data/task-1/preproc/1_original_train.bin"
    DATA_DIR = "../data/embeddings/"
    EMBED_FILE = os.path.join(DATA_DIR, "wiki-news-300d-1M.vec")

    m = None

    @classmethod
    def load_embeddings(cls):
        print("Building emebddings index")
        cls.FT = {}
        with open(cls.EMBED_FILE, "r") as fd:
            next(fd)
            for i, l in enumerate(fd):
                cls.FT[l.split(maxsplit=1)[0]] = i

    @classmethod
    def load_and_split_data(cls):
        if not cls.FT:
            cls.load_embeddings()

        raw_data, scores = [], []
        
        with open(cls.DATA_FILE, "rb") as fd:
            headlinecollection = lib.read_task1_pb(fd)
            for hl in headlinecollection:
                sent = hl.sentence
                sent[hl.word_index] = hl.edit
                if not hl.avg_grade:
                    score = 0.0
                else:
                    score = hl.avg_grade
                scores.append(score)
                raw_data.append(sent)

        raw_data = np.array(raw_data)
        scores = np.array(scores)

        proc_data = cls.process_raw_data(raw_data)

        return proc_data, scores

    @classmethod
    def process_raw_data(cls, raw_data):
        processed_data = []
        for sent in raw_data:
            sent_emb = np.zeros((300))
            for w in sent:
                try:
                    word_emb = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[w]).replace("\n", "").split(" ")[1:]])
                except KeyError:
                    word_emb = np.zeros((300))

                if word_emb.shape[0] != 300:
                    word_emb = np.zeros((300))
                sent_emb = np.add(sent_emb, word_emb)
            processed_data.append(sent_emb/len(sent))
        
        processed_data = np.array(processed_data, dtype="float64")
        return processed_data


    @classmethod
    def fit(cls, X, y):
        y = y.astype('float')
        cls.m = Sequential()
        cls.m.add(Dense(1, activation=sigmoid_3, input_dim=X.shape[1]))
        cls.m.compile(optimizer='adam', loss='mse', metrics=[metrics.RootMeanSquaredError()])
        cls.m.fit(X, y, nb_epoch=20, validation_split=0.2)

    @classmethod
    def evaluate(cls, X, y):
        print(cls.m.evaluate(X, y))
        

    @classmethod
    def predict(cls):
        all_sents = []
        all_raw_sents = []
        idxes = []
        with open("../data/task-1/preproc/1_original_dev.bin", "rb") as fd:
            headlinecollection = lib.read_task1_pb(fd)
            for hl in headlinecollection:
                idxes.append(hl.id)
                sent = hl.sentence
                sent[hl.word_index] = hl.edit
                all_raw_sents.append(sent)

        all_sents = cls.process_raw_data(all_raw_sents)

        predictions = cls.m.predict(all_sents)

        with open("../predictions/task-1-output-test.csv", 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['id','pred'])
            for i in range(len(predictions)):
                writer.writerow([idxes[i], predictions[i][0]])

