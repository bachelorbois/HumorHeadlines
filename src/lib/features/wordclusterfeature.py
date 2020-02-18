#import ..lib
from lib.parsing import Headline
from lib.features import Feature
from lib.features.embeddingContainer import EmbeddingContainer
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import wget
import linecache
from zipfile import ZipFile
import pickle


class ClusterFeature(Feature):
    MODEL_PATH = "./lib/models/kmeans/model.sav"

    @classmethod
    def train_kmeans(cls):
        print("Running kmeans")
        cls.kmeans = KMeans(n_clusters=20, init='k-means++', n_init=2, max_iter=10, random_state=42, tol=0.001, verbose=1)
        cls.kmeans.fit(cls.all_embeddings)
        pickle.dump(cls.kmeans, open(cls.MODEL_PATH, 'wb'))

    @classmethod
    def predict_kmeans(cls, words_emebdding_list):
        return cls.kmeans.predict(words_emebdding_list)

    @classmethod
    def compute_feature(cls, HL):
        EmbeddingContainer.init()

        replaced = HL.sentence[HL.word_index]
        replacement = HL.edit
        replaced_emb, replacement_emb = EmbeddingContainer.lookup(replaced, replacement)

        if not os.path.isfile(cls.MODEL_PATH):
            cls.train_kmeans()
        else:
            cls.kmeans = pickle.load(open(cls.MODEL_PATH, 'rb'))

        cls.preds = cls.predict_kmeans([replaced_emb, replacement_emb])

        return cls.preds
