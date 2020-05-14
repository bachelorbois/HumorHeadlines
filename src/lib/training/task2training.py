from tensorflow.keras import Model, callbacks
import tensorflow as tf
import numpy as np
from lib.parsing.parser import read_task2_pb
from lib.features import PhoneticFeature, PositionFeature, DistanceFeature, ClusterFeature, SentLenFeature, SarcasmFeature, NellKbFeature
from lib.features.embeddingContainer import EmbeddingContainer
import datetime
import os


class Task2Training:
    DIR = "./headline_prediction/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = DIR + '/logs/'
    SAVE_DIR = DIR + '/weights/'
    PRED_DIR = DIR + '/predictions/'
    VIZ_DIR = DIR + '/visualization/'
    PRED_FILE = PRED_DIR + '/task-2-output.csv'
    EMBED_FILE = "../data/embeddings/wiki-news-300d-1M"

    def __init__(self, task2model : Model, train_paths : List[str], dev_path : List[str], test_path : List[str]):
        self.model = task2model
        EmbeddingContainer.init()

        os.makedirs(self.LOG_DIR)
        os.makedirs(self.SAVE_DIR)
        os.makedirs(self.PRED_DIR)
        os.makedirs(self.VIZ_DIR)
        f = open(self.PRED_FILE, "w")
        f.close()

        tf.keras.utils.plot_model(
            self.humor, to_file=f'{self.VIZ_DIR}model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=True, dpi=96
        )

        self.train_data = self.load_data(train_paths)
        self.dev_data = self.load_data(dev_path)
        self.test_data = self.load_data(test_path)

        features = [PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature, NellKbFeature]

        self.train_data.AddFeatures(features)
        self.dev_data.AddFeatures(features)
        self.test_data.AddFeatures(features)

    def train(self, batch_size : int, epoch : int):
        # Train data
        ins, y_train = self.get_data_dict(self.train_data)
        # Dev data
        devIns, y_dev = self.get_data_dict(self.dev_data)

        # Create callbacks
        tensorboard = callbacks.TensorBoard(log_dir=self.LOG_DIR, write_graph=True, write_images=True)
        # lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_root_mean_squared_error', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0001)
        print(f"--------- Follow the training using Tensorboard at {self.LOG_DIR} ---------")
        self.model.fit(x=ins, y=y_train,
                        validation_data=(devIns, y_dev),
                        batch_size=batch_size,
                        epochs=epoch,
                        shuffle=True,
                        callbacks=[tensorboard])

        self.model.save(self.SAVE_DIR+'final.hdf5')

    def test(self):
        pass

    @staticmethod
    def load_data(paths):
        hc = None
        for path in paths:
            with open(path, 'rb') as fd:
                if not hc:
                    hc = read_task2_pb(fd)
                else:
                    hc.extend(read_task2_pb(fd))
        return hc

    @staticmethod
    def get_data_dict(data):
        features = data.GetFeatureVectors()
        HL1Features = features[:, 0, :4]
        HL2Features = features[:, 1, :4]

        HL1Entities = features[:, 0, 4:]
        HL2Entities = features[:, 1, 4:]

        HL1Tokens = np.array([c.HL1.edit for c in data])
        HL2Tokens = np.array([c.HL2.edit for c in data])

        HL1Sentences = np.array([h.HL1.sentence[h.HL1.word_index] for h in data])
        HL2Sentences = np.array([h.HL2.sentence[h.HL2.word_index] for h in data])

        ins = {"FeatureInputHL1": HL1Features,
                "EntityInputHL1": HL1Entities,
                "ReplacedInputHL1": HL1Sentences,
                "ReplacementInputHL1": HL1Tokens,
                "FeatureInputHL2": HL2Features,
                "EntityInputHL2": HL2Entities,
                "ReplacedInputHL2": HL2Sentences,
                "ReplacementInputHL2": HL2Tokens}

        labels = data.GetLabels()

        return ins, labels