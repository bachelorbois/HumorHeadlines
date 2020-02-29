from lib.features.embeddingContainer import EmbeddingContainer
from lib.parsing import Headline
from lib.features import Feature
import os
import pickle
from lib.models.sarcasm_FFNN import SarcasmClassifier


class SarcasmFeature(Feature):
    MODEL_PATH = "lib/models/pre-trained/sarcasm_model.h5"

    @classmethod
    def compute_feature(cls, HL):
        EmbeddingContainer.init()

        sentence = HL.sentence
        sentence[HL.word_index] = HL.edit

        cls.sc = SarcasmClassifier()

        if not os.path.isfile(cls.MODEL_PATH):
            EmbeddingContainer.BUILD_ALL = True
            EmbeddingContainer.init()
            cls.sc.run_preproc()
            cls.sc.run_model()
        else:
            cls.model = pickle.load(open(cls.MODEL_PATH, 'rb'))

        cls.processed_sentence = cls.sc.process_sentence(sentence)
        cls.preds = cls.predict_sarcasm([cls.processed_sentence])

        return cls.preds