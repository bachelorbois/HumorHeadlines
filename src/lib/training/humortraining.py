from tensorflow.keras import Model, callbacks
import tensorflow as tf
import numpy as np
import math
import datetime
import os
import pickle
from typing import List
import time
import matplotlib.pyplot as plt

from lib.models import create_HUMOR_model
from lib.parsing.parser import read_task1_pb
from lib.features import PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature, NellKbFeature, AlbertTokenizer
from lib.features.embeddingContainer import EmbeddingContainer
from lib.models.sarcasm_FFNN import SarcasmClassifier

class HumorTraining:
    DIR = "./headline_regression/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = DIR + '/logs/'
    SAVE_DIR = DIR + '/weights/'
    PRED_DIR = DIR + '/predictions/'
    VIZ_DIR = DIR + '/visualization/'
    PRED_FILE = PRED_DIR + '/task-1-output.csv'
    EMBED_FILE = "../data/embeddings/wiki-news-300d-1M"

    def __init__(self, Humor : Model, albert : bool, train_paths : List[str], dev_path : List[str], test_path : List[str]):
        self.start = time.time()
        self.humor = Humor
        self.albert = albert
        EmbeddingContainer.init()

        os.makedirs(self.LOG_DIR)
        os.makedirs(self.SAVE_DIR)
        os.makedirs(self.PRED_DIR)
        os.makedirs(self.VIZ_DIR)
        f = open(self.PRED_FILE, "w")
        f.close()

        #tf.keras.utils.plot_model(
        #    self.humor, to_file=f'{self.VIZ_DIR}model.png', show_shapes=True, show_layer_names=True,
        #    rankdir='TB', expand_nested=True, dpi=96
        #)

        self.train_data = self.load_data(train_paths)
        self.dev_data = self.load_data(dev_path)
        self.test_data = self.load_data(test_path)

        features = [PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature, NellKbFeature]

        if self.albert:
            features.append(AlbertTokenizer)

        self.train_data.AddFeatures(features)
        self.dev_data.AddFeatures(features)
        self.test_data.AddFeatures(features)

    def train(self, epoch, batch_size, validation_split=0.2):
        # Train data
        features, y_train = self.train_data.GetFeatureVectors(), self.train_data.GetGrades()
        ins = {"FeatureInput": features[:,:4]}
        i = 4
        ins["EntityInput"] = features[:,i:i+20]
        if self.albert:
            i += 20
            ins["input_word_ids"] = features[:,i:i+128]
            i += 128
            ins["segment_ids"] = features[:,i:i+128]
            i += 128
            ins["input_mask"] = features[:,i:i+128]

        text = np.load('../data/task-1/train_replaced.npy', allow_pickle=True)# self.train_data.GetReplaced()
        ins["ReplacedInput"] = text

        text = np.load('../data/task-1/train_edit.npy', allow_pickle=True)# self.train_data.GetEdits()
        ins["ReplacementInput"] = text

        # Dev data
        dev_features, y_dev = self.dev_data.GetFeatureVectors(), self.dev_data.GetGrades()
        devIns = {"FeatureInput": dev_features[:,:4]}
        i = 4
        devIns["EntityInput"] = dev_features[:,i:i+20]
        if self.albert:
            i += 20
            devIns["input_word_ids"] = dev_features[:,i:i+128]
            i += 128
            devIns["segment_ids"] = dev_features[:,i:i+128]
            i += 128
            devIns["input_mask"] = dev_features[:,i:i+128]

        text = np.load('../data/task-1/dev_replaced.npy', allow_pickle=True)# self.dev_data.GetReplaced()
        devIns["ReplacedInput"] = text

        text = np.load('../data/task-1/dev_edit.npy', allow_pickle=True)# self.dev_data.GetEdits()
        devIns["ReplacementInput"] = text

        # Create callbacks
        # tensorboard = callbacks.TensorBoard(log_dir=self.LOG_DIR, write_graph=True, write_images=True)
        lr_schedule = self.create_learning_rate_scheduler(max_learn_rate=1e-1,
                                                        end_learn_rate=1e-6,
                                                        warmup_epoch_count=15,
                                                        total_epoch_count=epoch)
        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, mode='min', restore_best_weights=True)
        # lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_root_mean_squared_error', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0001)
        print("Follow the training using Tensorboard at " + self.LOG_DIR)
        print(f"--------- It took {time.time() - self.start} second from start to training ---------")
        
        self.humor.fit(x=ins, y=y_train,
                        validation_data=(devIns, y_dev),
                        batch_size=batch_size,
                        epochs=epoch,
                        shuffle=True,
                        callbacks=[early])

        # preds = self.humor.predict(devIns)
        # ids = self.dev_data.GetIDs()
        # out = np.stack((ids, preds.flatten()), axis=-1)
        # Save the predictions to file
        # np.savetxt(f'../plots/context.csv', out, header='id,pred', fmt="%d,%1.8f")

        # print(f'Mean of preds: {preds.mean()}, STD of preds: {preds.std()}, Mean of true: {y_dev.mean()}, STD of true: {y_dev.std()}')
        # bins = np.linspace(0, 3, 50)
        # plt.hist(preds, bins=bins, alpha=0.5, label="preds")
        # plt.hist(y_dev, bins=bins, alpha=0.5, label="true")
        # plt.legend(loc='upper right')
        # plt.show()

        # self.humor.save(self.SAVE_DIR + 'final.h5', save_format='h5')

    def test(self):
        # Test data
        features = self.test_data.GetFeatureVectors()
        ins = {"FeatureInput": features[:,:4]}
        i = 4
        ins["EntityInput"] = features[:,i:i+20]
        if self.albert:
            i += 20
            ins["input_word_ids"] = features[:,i:i+128]
            i += 128
            ins["segment_ids"] = features[:,i:i+128]
            i += 128
            ins["input_mask"] = features[:,i:i+128]

        text = np.load('../data/task-1/test_replaced.npy', allow_pickle=True)# self.test_data.GetReplaced()
        ins["ReplacedInput"] = text
        text = np.load('../data/task-1/test_edit.npy', allow_pickle=True)# self.test_data.GetEdits()
        ins["ReplacementInput"] = text

        # Predict on the data
        preds = self.humor.predict(ins)
        ids = self.test_data.GetIDs()

        out = np.stack((ids, preds.flatten()), axis=-1)
        # Save the predictions to file
        np.savetxt(self.PRED_FILE, out, header='id,pred', fmt="%d,%1.8f")

    @staticmethod
    def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                        end_learn_rate=1e-7,
                                        warmup_epoch_count=10,
                                        total_epoch_count=90):

        def lr_scheduler_exp_decay(epoch):
            if epoch < warmup_epoch_count:
                res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
            else:
                res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
            return float(res)

        def lr_scheduler_step_decay(epoch):
            initial_lrate = 0.005
            drop = 0.5
            epochs_drop = 10.0
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_step_decay, verbose=1)

        return learning_rate_scheduler

    @staticmethod
    def load_data(paths):
        hc = None
        for path in paths:
            with open(path, 'rb') as fd:
                if not hc:
                    hc = read_task1_pb(fd)
                else:
                    hc.extend(read_task1_pb(fd))
        return hc
