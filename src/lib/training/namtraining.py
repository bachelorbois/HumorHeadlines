import datetime
import os
import random

from sklearn.model_selection import train_test_split
import numpy as np
from rdflib import Graph
from tensorflow.keras import callbacks
from tqdm import tqdm

class NAMTraining:
    DIR = "./entity_representation/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = DIR + '/logs/'
    SAVE_DIR = DIR + '/weights/'
    DATA_DIR = '../data/NELL/'
    PREPROC_DIR = DATA_DIR + 'preproc/'
    EMBED_DIR = DATA_DIR + 'embeddings/'
    DATAX = PREPROC_DIR + 'triples.npy'
    DATAY = PREPROC_DIR + 'labels.npy'
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
        reduceLR = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0.0001)

        self.model.fit([self.train_x[:, 0], self.train_x[:, 1], self.train_x[:, 2]], self.train_y, 
                        epochs=epoch, 
                        batch_size=batch_size, 
                        validation_split=.2,
                        shuffle=True,
                        callbacks=[tensorboard, reduceLR])

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
        rel_vocab['UNK'] = 0
        with open(entity_vocab, 'r') as f:
            for i, line in enumerate(f):
                ent_vocab[line.strip()] = i+1

        with open(relation_vocab, 'r') as f:
            for i, line in enumerate(f):
                rel_vocab[line.strip()] = i+1
        print("Parsed the dictionaries...")
        if not os.path.isfile(self.DATAX):
            g = Graph()
            with open(path, 'r') as f:
                g.parse(data=f.read(), format="xml")
            print("Parsed the NELL Graph...")
            print(f"No. points in graph: {len(g)}")
            for e1, r, e2 in tqdm(g):
                if r.strip() != "concept:haswikipediaurl" and r.strip() != "concept:latitudelongitude":
                    if f'{e1}' in list(ent_vocab) and f'{e2}' in list(ent_vocab) and f'{r}' in list(rel_vocab):
                        triple1 = np.array([ent_vocab[f'{e1}'], rel_vocab[f'{r}'], ent_vocab[f'{e2}']])
                        data.append(triple1)
                        y.append(1)
                        new_e1 = self.pick_random(g, e1, r, e2, ent_vocab)
                        triple2 = np.array([ent_vocab[new_e1], rel_vocab[f'{r}'], ent_vocab[f'{e2}']])
                        data.append(triple2)
                        y.append(0)
                    # Generalizations
                    e1_list = e1.split(':')
                    e2_list = e2.split(':')
                    if f'{e1_list[0]}:{e1_list[1]}' in list(ent_vocab):
                        triple1 = np.array([ent_vocab[f'{e1}'], rel_vocab['concept:isa'], ent_vocab[f'{e1_list[0]}:{e1_list[1]}']])
                        data.append(triple1)
                        y.append(1)
                        new_e1 = self.pick_random_2(e1, e1_list, ent_vocab)
                        triple2 = np.array([ent_vocab[new_e1], rel_vocab['concept:isa'], ent_vocab[f'{e1_list[0]}:{e1_list[1]}']])
                        data.append(triple2)
                        y.append(0)

                    if f'{e2_list[0]}:{e2_list[1]}' in list(ent_vocab):
                        triple1 = np.array([ent_vocab[f'{e2}'], rel_vocab['concept:isa'], ent_vocab[f'{e2_list[0]}:{e2_list[1]}']])
                        data.append(triple1)
                        y.append(1)
                        new_e2 = self.pick_random_2(e2, e2_list, ent_vocab)
                        triple2 = np.array([ent_vocab[new_e2], rel_vocab['concept:isa'], ent_vocab[f'{e2_list[0]}:{e2_list[1]}']])
                        data.append(triple2)
                        y.append(0)   
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

    def pick_random_2(self, e, e_list, vocab):
        new_e = e
        while new_e == e or f'{new_e}' == f'{e_list[0]}:{e_list[1]}':
            new_e = random.sample(list(vocab), 1)[0]

        return new_e