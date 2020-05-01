from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from lib import parsing
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn

class Probing():
    model = None
    triples, labels, entity_emb = None, None, None
    X, y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None

    def load_data(self):
        self.triples = np.load("../data/NELL/preproc/triples.npy")
        self.labels = np.load("../data/NELL/preproc/labels.npy")
        self.entity_emb = np.load("../data/NELL/embeddings/entity.npy")
        
        """
        relation_emb = np.load("../data/NELL/embeddings/relation.npy")
        
        vocab = []
        with open("../data/NELL/NELLWordNetVocab_proc.txt") as infile:
            for line in infile:
                vocab.append(line.strip().split(":")[1])
        vocab = np.array(vocab)
        
        rel_vocab = []
        with open("../data/NELL/NELLRelVocab.txt") as infile:
            for line in infile:
                rel_vocab.append(line.strip().split(":")[1])
        rel_vocab = np.array(rel_vocab)
        """

    def filter_relations(self):
        counter = Counter(self.triples[:,1])
        selected_relations_type = []
        rel_new_idx = {}
        idx = 0
        s = 0
        for elem in counter.keys():
            if counter[elem] >= 700 and counter[elem] <= 1300:
                s += counter[elem]
                selected_relations_type.append(elem)
                rel_new_idx[elem] = idx
                idx += 1
        
        self.X, self.y = [], []
        for idx, triple in enumerate(self.triples):
            if triple[1] in selected_relations_type and self.labels[idx] == 1:
                emb_e1 = self.entity_emb[triple[0]]
                emb_e2 = self.entity_emb[triple[2]]
                self.X.append([emb_e1, emb_e2])
                self.y.append(rel_new_idx[triple[1]])


    def train_test(self):
        #print(s)
        #print(idx)

        y_classes = np.array(self.y)
        self.y = to_categorical(y_classes)
                
        self.X = np.array(self.X)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1]*self.X.shape[2]))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=0.20)

    def run_model(self):
        input_layer = Input(shape=(self.X.shape[1],))
        hidden1 = Dense(256, activation='relu')(input_layer)
        hidden2 = Dense(128, activation='relu')(hidden1)
        output = Dense(self.y.shape[1], activation='softmax')(hidden2)
        
        self.model = Model(inputs=input_layer, outputs=output)

        print(self.model.summary())

        self.model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=10, validation_split=0.10)

        self.model.evaluate(self.X_test, self.y_test)


    def predict_and_plot(self):
        preds = np.argmax(self.model.predict(self.X_test), axis=1)

        correct = 0
        for i in range(len(self.y_test)):
            if np.argmax(self.y_test[i]) == preds[i]:
                correct += 1

        print("Accuracy: ", correct/len(self.X_test))
        
        cm = confusion_matrix(np.argmax(self.y_test, axis=1), preds)

        df_cm = pd.DataFrame(cm)#, index = [i for i in range(99)], columns = [i for i in range(99)])
        plt.figure(figsize = (12,10))
        sn.heatmap(df_cm, annot=False)
        plt.xlabel("Gold")
        plt.ylabel("Prediction")
        plt.title("Confusion Matrix of Relation Classifier")

        plt.savefig("lib/probing/entity_cm.png")

    def execute(self):
        self.load_data()
        self.filter_relations()
        self.train_test()
        self.run_model()
        self.predict_and_plot()
