from tensorflow.keras import Model, callbacks
import tensorflow as tf
import numpy as np
import math
import datetime
import os

from lib.models import create_HUMOR_model
from lib.parsing.parser import read_task1_pb
from lib.features import PhoneticFeature, PositionFeature, DistanceFeature, ClusterFeatures, SentLenFeature

class HumorTraining:
    DIR = "./headline_regression/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = DIR + '/logs/'
    SAVE_DIR = DIR + '/weights/'
    PRED_DIR = DIR + '/predictions/'
    PRED_FILE = PRED_DIR + '/task-1.txt'

    def __init__(self, Humor : Model, embeds : bool, train_path : str, test_path : str):
        self.humor = Humor
        self.embeds = embeds

        os.makedirs(self.LOG_DIR)
        os.makedirs(self.SAVE_DIR)
        os.makedirs(self.PRED_DIR)
        os.mknod(self.PRED_FILE)

        self.train_data = self.load_data(train_path)
        self.test_data = self.load_data(test_path)

        features = [PhoneticFeature, PositionFeature, DistanceFeature, ClusterFeatures, SentLenFeature]

        self.train_data.AddFeatures(features)

    def train(self, epoch, batch_size, validation_split=0.2):
        features, y_train = self.train_data.GetFeatureVectors(), self.train_data.GetGrades()
        ins = {"feature_input": features}

        if (self.embeds):
            text = self.train_data.GetEditSentences()
            ins = {"feature_input": features, "string_input": text}

        # Create callbacks
        tensorboard = callbacks.TensorBoard(log_dir=self.LOG_DIR)
        lr_schedule = self.create_learning_rate_scheduler(max_learn_rate=1e-3,
                                                        end_learn_rate=1e-7,
                                                        warmup_epoch_count=5,
                                                        total_epoch_count=epoch)

        print("Follow the training using Tensorboard at " + self.LOG_DIR)

        self.humor.fit(x=ins, y=y_train,
                        validation_split=validation_split,
                        batch_size=batch_size,
                        epochs=epoch,
                        shuffle=True,
                        callbacks=[lr_schedule])

    def test(self):
        # Test data
        features = self.test_data.GetFeatureVectors()
        ins = {"feature_input": features}

        if (self.embeds):
            text = self.test_data.GetEditSentences()
            ins = {"feature_input": features, "string_input": text}
        # Predict on the data
        preds = self.humor.predict(ins)

        # Save the predictions to file
        np.savetxt(self.PRED_FILE, preds)

    @staticmethod
    def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

        def lr_scheduler(epoch):
            if epoch < warmup_epoch_count:
                res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
            else:
                res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
            return float(res)
        
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

        return learning_rate_scheduler
    
    @staticmethod
    def load_data(path):
        with open(path, 'rb') as fd:
            data = read_task1_pb(fd)

        return data