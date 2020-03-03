import os
import wget
import linecache
from zipfile import ZipFile
from typing import Tuple
import numpy as np
from tqdm import tqdm

class EmbeddingContainer():
    """A static container class managing embedding allocation.
    """
    FT = None
    ALL = None

    BUILD_ALL = False

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

            for i, l in tqdm(enumerate(fd)):
                cls.FT[str.split(l, maxsplit=1)[0]] = i

    @classmethod
    def load_all(cls) -> None:
        print("Building full embedding list...")

        cls.ALL = []
        for w in cls.FT.keys():
            cls.key_emb = np.array([float(e) for e in linecache.getline(cls.EMBED_FILE, cls.FT[w]).replace("\n", "").split(" ")[1:]])
            if cls.key_emb.shape[0] == 300:
                cls.ALL.append(cls.key_emb)

        cls.ALL = np.array(cls.ALL)

    @classmethod
    def lookup_single(cls, word : str) -> np.ndarray:
        """Looks up a single word in the embedding

        Args:
            word (str): A word to look up

        Returns:
            np.ndarray: A 300d embedding vector
        """
        try:
            return np.array([float(e) for e in str.split(str.replace(linecache.getline(cls.EMBED_FILE, cls.FT[word]), "\n", ""), " ")[1:]])
        except:
            return np.zeros((300))

    @classmethod
    def lookup(cls, replaced : str, replacement : str) -> Tuple[np.ndarray, np.ndarray]:
        """Looks up the appropriate embedding for a replaced and replacement word

        Args:
            replaced (str): A word
            replacement (str): A word

        Returns:
            (np.ndarray, np.ndarray): An embedding vector for each word
        """
        return cls.lookup_single(replaced), cls.lookup_single(replacement)

    @classmethod
    def init(cls) -> None:
        """Ensures that the container is properly initialised. Must be run before doing a lookup to ensure proper configuration.
        """
        if cls.FT is None:
            cls.load_ft()

        if cls.BUILD_ALL and cls.ALL is None:
            cls.load_all()
