from tensorflow.keras import Model, callbacks
import tensorflow as tf
import numpy as np
import math
import datetime
import os
import pickle

from lib.models import create_HUMOR_model
from lib.parsing.parser import read_task1_pb
from lib.features import PhoneticFeature, PositionFeature, DistanceFeature, ClusterFeature, SentLenFeature

class HumorTraining:
    DIR = "./headline_regression/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = DIR + '/logs/'
    SAVE_DIR = DIR + '/weights/'
    PRED_DIR = DIR + '/predictions/'
    PRED_FILE = PRED_DIR + '/task-1-output.csv'
    EMBED_FILE = "../data/embeddings/wiki-news-300d-1M"

    def __init__(self, Humor : Model, embeds : bool, train_path : str, test_path : str):
        self.humor = Humor
        self.embeds = embeds
        print("Loading fastText Embedings...")
        self.fastTextEmbeds = self.load_embeddings()

        os.makedirs(self.LOG_DIR)
        os.makedirs(self.SAVE_DIR)
        os.makedirs(self.PRED_DIR)
        os.mknod(self.PRED_FILE)

        self.train_data = self.load_data(train_path)
        self.test_data = self.load_data(test_path)

        features = [PhoneticFeature, PositionFeature, DistanceFeature, ClusterFeature, SentLenFeature]

        self.train_data.AddFeatures(features)
        self.test_data.AddFeatures(features)

    def train(self, epoch, batch_size, validation_split=0.2):
        features, y_train = self.train_data.GetFeatureVectors(), self.train_data.GetGrades()
        ins = {"feature_input": features}

        token = self.train_data.GetTokenizedWEdit()
        processedSents = self.process_sentences(token, self.fastTextEmbeds)
        ins["token_input"] = processedSents

        if (self.embeds):
            text = self.train_data.GetEditSentences()
            ins["string_input"] = text

        # Create callbacks
        tensorboard = callbacks.TensorBoard(log_dir=self.LOG_DIR)
        lr_schedule = self.create_learning_rate_scheduler(max_learn_rate=1e-2,
                                                        end_learn_rate=1e-6,
                                                        warmup_epoch_count=20,
                                                        total_epoch_count=epoch)

        print("Follow the training using Tensorboard at " + self.LOG_DIR)

        self.humor.fit(x=ins, y=y_train,
                        validation_split=validation_split,
                        batch_size=batch_size,
                        epochs=epoch,
                        shuffle=True,
                        callbacks=[lr_schedule, tensorboard])

        self.humor.save(self.SAVE_DIR+'final.hdf5')

    def test(self):
        # Test data
        features = self.test_data.GetFeatureVectors()
        ins = {"feature_input": features}

        token = self.test_data.GetTokenizedWEdit()
        processedSents = self.process_sentences(token, self.fastTextEmbeds)
        ins["token_input"] = processedSents

        if (self.embeds):
            text = self.test_data.GetEditSentences()
            ins["string_input"] = text

        # Predict on the data
        preds = self.humor.predict(ins)
        ids = self.test_data.GetIDs()
        
        out = np.stack((ids, preds.flatten()), axis=-1)
        # Save the predictions to file
        np.savetxt(self.PRED_FILE, out, fmt="%d,%1.8f")

    @staticmethod
    def process_sentences(tokenized_sentences, dictionary):
        proc_sentences = [] 
        for sentence in tokenized_sentences:
            agg = np.zeros((300))
            for token in sentence:
                try:
                    agg = np.add(agg, dictionary[token])
                except KeyError:
                    agg = np.add(agg, dictionary['UNK'])
            proc_sentences.append(agg/len(sentence))

        return np.array(proc_sentences)

    def load_embeddings(self):
        dictionary = {}
        if not os.path.isfile(self.EMBED_FILE + '.p'):
            with open(self.EMBED_FILE + '.vec', 'r') as infile:
                for line in infile:
                    line = line.split()
                    word = line[0]
                    emb = np.array(line[1:], dtype='float')
                    dictionary[word] = emb

            dictionary['UNK'] = np.array(list(dictionary.values())).mean()

            with open(self.EMBED_FILE + '.p', 'wb') as f:
                pickle.dump(dictionary, f)
        else:
            with open(self.EMBED_FILE + '.p', 'rb') as f:
                dictionary = pickle.load(f)
        
        return dictionary

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
            epochs_drop = 5.0
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
        
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_step_decay, verbose=1)

        return learning_rate_scheduler
    
    @staticmethod
    def load_data(path):
        with open(path, 'rb') as fd:
            data = read_task1_pb(fd)

        return data