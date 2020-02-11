import lib
from lib.parsing import Headline
from lib.features import Feature
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import silhouette_score
from fcmeans import FCM
import matplotlib.pyplot as plt
import os
import wget
import linecache
from zipfile import ZipFile
import pickle

DATA_FILE = "data/task-1/preproc/1_original_train.bin"


class ClusterFeatures(Feature):

    FT = None

    DATASET_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    DATA_DIR = "data/embeddings/"
    EMBED_FILE = os.path.join(DATA_DIR, "wiki-news-300d-1M.vec")
    MODEL_PATH = "src/lib/models/kmeans/model.sav"

    @classmethod
    def load_embeddings(cls):
        print("Building embedding index...")
        cls.FT = {}
        with open(cls.EMBED_FILE, "r") as fd:
            next(fd)
            for i, l in enumerate(fd):
                cls.FT[l.split(maxsplit=1)[0]] = i

        if not os.path.isfile(cls.MODEL_PATH):
            cls.all_embeddings = []
            for w in cls.FT.keys():
                cls.key_emb = []
                cls.key_emb = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[w]).replace("\n", "").split(" ")[1:]])
                if cls.key_emb.shape[0] == 300:
                    cls.all_embeddings.append(cls.key_emb)

            cls.all_embeddings = np.array(cls.all_embeddings)
        
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
        if cls.FT is None:
            cls.load_embeddings()

        replaced = HL.sentence[HL.word_index]
        replacement = HL.edit
        replaced_emb, replacement_emb = cls.word_embeddings_lookup(replaced, replacement)

        if not os.path.isfile(cls.MODEL_PATH):
            cls.train_kmeans()
        else:
            cls.kmeans = pickle.load(open(cls.MODEL_PATH, 'rb'))

        cls.preds = cls.predict_kmeans([replaced_emb, replacement_emb])
        
        return cls.preds
        
    @classmethod
    def word_embeddings_lookup(cls, replaced, replacement):
        try:
            replaced_emb = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[replaced]).replace("\n", "").split(" ")[1:]])
        except KeyError:
            replaced_emb = np.zeros((300))
        
        try:
            replacement_emb = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[replacement]).replace("\n", "").split(" ")[1:]])
        except KeyError:
            replacement_emb = np.zeros((300))

        return replaced_emb, replacement_emb
    

"""
if __name__ == "__main__":
    
    c = ClusterFeatures

    with open(DATA_FILE, "rb") as fd:
        headlinecollection = lib.read_task1_pb(fd)

        headlinecollection.AddFeature(lib.WordClustersFeature)
        all_feats = []
        for hl in headlinecollection:
            feat = c.compute_feature(hl)
            all_feats.append(feat)
"""