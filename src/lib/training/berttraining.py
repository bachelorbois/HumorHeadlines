# Standard libraries
import datetime

# Third-party libraries
import tensorflow as tf
from tensorflow.keras import Model, callbacks
import numpy as np
np.set_printoptions(threshold=np.inf)

# Custom libraries
from lib.parsing import Headline, read_task1_pb
import os
import math



class BertTraining:
    DIR = "./headline_regression/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = DIR + '/logs/'
    SAVE_DIR = DIR + '/weights/'
    PRED_DIR = DIR + '/predictions/'
    PRED_FILE = PRED_DIR + '/task-1.txt'
    BERT_VOCAB = f'{os.getcwd()}/lib/models/pre-trained/vocab.txt'

    def __init__(self, bertie : Model, train_path : str, test_path : str):
        # The given model (Presumably a BERT model)
        self.bertie = bertie

        os.makedirs(self.LOG_DIR)
        os.makedirs(self.SAVE_DIR)
        os.makedirs(self.PRED_DIR)
        os.mknod(self.PRED_FILE)
        
        # Load the vocab
        Headline.BERT_VECTOR_LENGTH = 30
        with open(self.BERT_VOCAB, 'r') as fd:
            Headline.SetBERTVocab(fd)

        # Loading the training
        self.train_data = self.load_data(train_path)
        self.test_data = self.load_data(test_path)

    def train(self, epoch, batch_size, validation_split=0.2):
        # Run the model please
        x_train, y_train = self.train_data.GetBERT(), tf.convert_to_tensor(self.train_data.GetGrades(), dtype=tf.float32)
        y_train = tf.where(tf.math.is_nan(y_train), tf.ones_like(y_train) * 0, y_train)

        # Create callbacks
        tensorboard = callbacks.TensorBoard(log_dir=self.LOG_DIR)
        saver = callbacks.ModelCheckpoint(filepath=self.SAVE_DIR+'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        lr_schedule = self.create_learning_rate_scheduler(max_learn_rate=1e-6,
                                                        end_learn_rate=1e-9,
                                                        warmup_epoch_count=int(epoch * 0.1),
                                                        total_epoch_count=epoch)

        # Train the model (Takes a long time)
        self.bertie.fit(x=x_train, y=y_train,
                        validation_split=validation_split,
                        batch_size=batch_size,
                        epochs=epoch,
                        shuffle=True,
                        callbacks=[tensorboard, saver, lr_schedule])

        # Save the final weights
        self.bertie.save_weights(self.SAVE_DIR+'final.hdf5')

    def test(self):
        # Test data
        x_test = self.test_data.GetBERT()

        # Predict on the data
        preds = self.bertie.predict(x_test)

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