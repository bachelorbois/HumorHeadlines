from lib.features.embeddingContainer import EmbeddingContainer
from lib.parsing import Headline
from lib.features import Feature
import os
import pickle
from lib.models.sarcasm_FFNN import SarcasmClassifier


class SarcasmFeature(Feature):
    MODEL_PATH = "lib/models/pre-trained/sarcasm_model.h5"
    model = None

    @classmethod
    def compute_feature(cls, HL):
        EmbeddingContainer.init()

        sentence = HL.sentence
        sentence[HL.word_index] = HL.edit
        if not cls.model:
            cls.model = SarcasmClassifier()
            if os.path.isfile(cls.MODEL_PATH):
                cls.model.load_model(cls.MODEL_PATH)
            else:
                EmbeddingContainer.BUILD_ALL = True
                EmbeddingContainer.init()
                cls.model.run_preproc()
                cls.model.run_model()
        
        cls.processed_sentence = cls.model.process_sentence(sentence)
        cls.preds = cls.model.predict_sarcasm(cls.processed_sentence)

        return cls.preds