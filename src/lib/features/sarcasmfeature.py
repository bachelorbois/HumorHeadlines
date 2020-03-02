from lib.features.embeddingContainer import EmbeddingContainer
from lib.parsing import Headline
from lib.features import Feature
import os
import pickle
import numpy as np
from lib.models.sarcasm_FFNN import SarcasmClassifier


class SarcasmFeature(Feature):
    MODEL_PATH = "lib/models/pre-trained/sarcasm_model.h5"

    @classmethod
    def compute_feature(cls, HL):
        EmbeddingContainer.init()

        sentence = HL.sentence
        sentence[HL.word_index] = HL.edit
        if os.path.isfile(cls.MODEL_PATH):
            SarcasmClassifier.load_model(cls.MODEL_PATH)
        else:
            SarcasmClassifier.run_preproc()
            SarcasmClassifier.run_model()
        
        cls.processed_sentence = SarcasmClassifier.process_sentence(sentence)
        cls.preds = SarcasmClassifier.predict_sarcasm(cls.processed_sentence)

        return cls.preds