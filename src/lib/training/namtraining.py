import datetime
import os
import random

from sklearn.model_selection import train_test_split
import numpy as np
from rdflib import Graph
from tensorflow.keras import callbacks

class NAMTraining:
    DIR = "./entity_representation/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = DIR + '/logs/'
    SAVE_DIR = DIR + '/weights/'
    DATA_DIR = '../data/NELL/'
    PREPROC_DIR = DATA_DIR + 'preproc/'
    EMBED_DIR = DATA_DIR + 'embeddings/'
    DATAX = DATA_DIR + 'triples.npy'
    DATAY = DATA_DIR + 'labels.npy'
    REL_EMBED = EMBED_DIR + 'relation.npy'
    ENT_EMBED = EMBED_DIR + 'entity.npy'
    def __init__(self, model, data_path, entity_vocab, relation_vocab):
        self.model = model
        self.data_x, self.data_y = self.load_data(data_path, entity_vocab, relation_vocab)

        os.makedirs(self.LOG_DIR)
        os.makedirs(self.SAVE_DIR)
        os.makedirs(self.PREPROC_DIR, exist_ok=True)
        os.makedirs(self.EMBED_DIR, exist_ok=True)

        np.save(self.DATAX, self.data_x)
        np.save(self.DATAY, self.data_y)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data_x, self.data_y, test_size=0.2, random_state=42)
        print(f'Train X : {self.train_x.shape} \n')
        print(f'Train Y : {self.train_y.shape} \n')
        print(f'Test X : {self.test_x.shape} \n')
        print(f'Test Y : {self.test_y.shape} \n')

    def train(self, epoch, batch_size):
        print("Training the model...")

        tensorboard = callbacks.TensorBoard(log_dir=self.LOG_DIR)

        self.model.fit([self.train_x[:, 0], self.train_x[:, 1], self.train_x[:, 2]], self.train_y, 
                        epochs=epoch, 
                        batch_size=batch_size, 
                        validation_split=.2,
                        shuffle=True,
                        callbacks=[tensorboard])

        self.model.save(self.SAVE_DIR+'final.hdf5')

    def test(self):
        self.model.evaluate([self.test_x[:, 0], self.test_x[:, 1], self.test_x[:, 2]], self.test_y)

        ent_embed = self.model.layers[2].get_weights()[0]
        rel_embed = self.model.layers[3].get_weights()[0]

        np.save(self.ENT_EMBED, ent_embed)
        np.save(self.REL_EMBED, rel_embed)

    def load_data(self, path, entity_vocab, relation_vocab):
        ent_vocab, rel_vocab = {}, {}
        data = []
        y = []
        ent_vocab['UNK'] = 0
        with open(entity_vocab, 'r') as f:
            for i, line in enumerate(f):
                ent_vocab[line.strip()] = i+1

        with open(relation_vocab, 'r') as f:
            for i, line in enumerate(f):
                rel_vocab[line.strip()] = i
        print("Parsed the dictionaries...")
        if not os.path.isfile(self.DATAX):
            g = Graph()
            with open(path, 'r') as f:
                g.parse(data=f.read(), format="xml")
            print("Parsed the NELL Graph...")
            print(f"No. points in graph: {len(g)}")
            idx = 0
            sizes = [int(len(g)*(i/10)) for i in range(10)]
            print("Will report when we have finished: ", sizes)
            for e1, r, e2 in g:
                if r.strip() != "concept:haswikipediaurl" and r.strip() != "concept:generalizations" and r.strip() != "concept:latitudelongitude":
                    if f'{e1}' in list(ent_vocab) and f'{e2}' in list(ent_vocab) and f'{r}' in list(rel_vocab):
                        triple1 = np.array([ent_vocab[f'{e1}'], rel_vocab[f'{r}'], ent_vocab[f'{e2}']])
                        data.append(triple1)
                        y.append(1)
                        new_e1 = self.pick_random(g, e1, r, e2, ent_vocab)
                        triple2 = np.array([ent_vocab[new_e1], rel_vocab[f'{r}'], ent_vocab[f'{e2}']])
                        data.append(triple2)
                        y.append(0)
                idx += 1
                if idx in sizes:
                    print(f'Currently at {idx}')
        else:
            print("Data file exists...")
            data = np.load(self.DATAX)
            y = np.load(self.DATAY)
        print("Created the training data...")
        return np.array(data), np.array(y)

    def pick_random(self, g, e1, r, e2, vocab):
        new_e = e1
        while (new_e, r, e2) in g:
            new_e = random.sample(list(vocab), 1)[0]

        return new_e