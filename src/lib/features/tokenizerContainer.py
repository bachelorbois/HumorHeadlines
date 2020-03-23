import tensorflow_hub as hub
from lib.tokenizer import FullSentencePieceTokenizer

class TokenizerContainer:
    TOKENIZER = None
    SP_MODEL = None

    @classmethod
    def load_tokenizer(cls):
        cls.SP_MODEL = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/1", trainable=False).resolved_object.sp_model_file.asset_path.numpy()
        cls.TOKENIZER = FullSentencePieceTokenizer(cls.SP_MODEL)

    @classmethod
    def init(cls):
        if cls.TOKENIZER is None:
            cls.load_tokenizer()