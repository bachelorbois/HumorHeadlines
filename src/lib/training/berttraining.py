# Standard libraries
import datetime

# Third-party libraries
import tensorflow as tf
from tensorflow.keras import Model, callbacks

# Custom libraries
from lib.parsing import Headline, read_task1_pb
from lib.models.bertmodel import create_model
import os



class BertTraining:
    DIR = "./headline_regression/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
    LOG_DIR = DIR + '/logs/'
    SAVE_DIR = DIR + '/weights/'
    BERT_VOCAB = f'{os.getcwd()}/lib/models/pre-trained/vocab.txt'

    def __init__(self, bertie : Model, train_path : str, test_path : str):
        # The given model (Presumably a BERT model)
        self.bertie = bertie
        
        # Load the vocab
        Headline.BERT_VECTOR_LENGTH = 30
        with open(self.BERT_VOCAB, 'r') as fd:
            Headline.SetBERTVocab(fd)

        # Loading the training
        self.train_data = self.load_data(train_path)

    def train(self, epoch, batch_size, validation_split=0.2):
        # Run the model please
        x_train, y_train = self.train_data.GetBERT(), tf.convert_to_tensor(self.train_data.GetGrades(), dtype=tf.float32)

        tensorboard = callbacks.TensorBoard(log_dir=self.LOG_DIR)
        saver = callbacks.ModelCheckpoint(filepath=self.SAVE_DIR+'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

        self.bertie.fit(x=x_train, y=y_train,
                        validation_split=validation_split,
                        batch_size=batch_size,
                        epochs=epoch,
                        shuffle=True,
                        callbacks=[tensorboard, saver])

        self.bertie.save_weights(self.SAVE_DIR+'final.hdf5')

    @staticmethod
    def load_data(path):
        with open(path, 'rb') as fd:
            data = read_task1_pb(fd)

        return data