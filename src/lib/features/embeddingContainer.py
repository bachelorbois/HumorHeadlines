import os
import wget
import linecache
from zipfile import ZipFile
import numpy as np

class EmbeddingContainer():
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
    def lookup(cls, replaced, replacement):
        replaced_emb, replacement_emb = np.zeros((300)), np.zeros((300))
        try:
            replaced_emb = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[replaced]).replace("\n", "").split(" ")[1:]])
        except KeyError:
            replaced_emb = np.zeros((300))

        try:
            replacement_emb = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[replacement]).replace("\n", "").split(" ")[1:]])
        except KeyError:
            replacement_emb = np.zeros((300))

        return replaced_emb, replacement_emb

    @classmethod
    def init(cls) -> None:
        if cls.FT is None:
            cls.load_ft()
