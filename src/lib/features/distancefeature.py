import os
import wget
import linecache
from zipfile import ZipFile
import numpy as np
from lib.parsing import Headline
from lib.features import Feature

class DistanceFeature(Feature):
    FT = None

    DATASET_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    DATA_DIR = "../data/embeddings/"
    EMBED_FILE = os.path.join(DATA_DIR, "wiki-news-300d-1M.vec")

    @classmethod
    def download_ft(cls) -> None:
        print("Downloading embeddings...")

        if not os.path.isdir(cls.DATA_DIR):
            os.mkdir(cls.DATA_DIR)

        zip_path = os.path.join(cls.DATA_DIR, "embed.zip")

        wget.download(cls.DATASET_URL, zip_path)
        print("")

        with ZipFile(zip_path, "r") as zipfile:
            zipfile.extractall(cls.DATA_DIR)

        os.remove(zip_path)

    @classmethod
    def load_ft(cls) -> None:
        if not os.path.isfile(cls.EMBED_FILE):
            cls.download_ft()

        print("Building embedding index...")
        cls.FT = {}
        with open(cls.EMBED_FILE, "r") as fd:
            next(fd)

            for i, l in enumerate(fd):
                cls.FT[l.split(maxsplit=1)[0]] = i


    @classmethod
    def compute_feature(cls, HL : Headline) -> np.ndarray:
        if cls.FT is None:
            cls.load_ft()

        w1 = HL.sentence[HL.word_index]
        w2 = HL.edit

        if w1 not in cls.FT or w2 not in cls.FT:
            return np.array([0])

        e1 = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[w1]).replace("\n", "").split(" ")[1:]])
        e2 = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[w2]).replace("\n", "").split(" ")[1:]])

        cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

        return np.array([cos])
