import datetime
import os
import random
import threading
import itertools

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

        # print(self.data_x, self.data_y)

        os.makedirs(self.LOG_DIR)
        os.makedirs(self.SAVE_DIR)
        os.makedirs(self.PREPROC_DIR, exist_ok=True)
        os.makedirs(self.EMBED_DIR, exist_ok=True)

        np.save(self.DATAX, self.data_x)
        np.save(self.DATAY, self.data_y)

        # print(self.data_x.shape)
        # print(self.data_y.shape)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data_x, self.data_y, test_size=0.2, random_state=42)
        #print(f'Train X : {self.train_x.shape} \n')
        #print(f'Train Y : {self.train_y.shape} \n')
        #print(f'Test X : {self.test_x.shape} \n')
        #print(f'Test Y : {self.test_y.shape} \n')

    def train(self, epoch, batch_size):
        print("Training the model...")

        # tensorboard = callbacks.TensorBoard(log_dir=self.LOG_DIR)
        reduceLR = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
        early = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, mode='auto', restore_best_weights=True)

        self.model.fit([self.train_x[:, 0], self.train_x[:, 1], self.train_x[:, 2]], self.train_y, 
                        epochs=epoch, 
                        batch_size=batch_size, 
                        validation_split=.2,
                        shuffle=True,
                        callbacks=[reduceLR, early])

        self.model.save(self.SAVE_DIR+'final.hdf5')

    def test(self):
        self.model.evaluate([self.test_x[:, 0], self.test_x[:, 1], self.test_x[:, 2]], self.test_y)

        ent_embed = self.model.layers[2].get_weights()[0]
        rel_embed = self.model.layers[3].get_weights()[0]

        np.save(self.ENT_EMBED, ent_embed)
        np.save(self.REL_EMBED, rel_embed)

    def load_data(self, path, entity_vocab, relation_vocab):
        self.ent_vocab, self.rel_vocab = {}, {}
        self.ent_vocab['UNK'] = 0
        self.rel_vocab['UNK'] = 0
        with open(entity_vocab, 'r') as f:
            for i, line in enumerate(f):
                self.ent_vocab[line.strip()] = i+1

        with open(relation_vocab, 'r') as f:
            for i, line in enumerate(f):
                self.rel_vocab[line.strip()] = i+1
        print("Parsed the dictionaries...")
        if not os.path.isfile(self.DATAX):
            self.graph = Graph()
            with open(path, 'r') as f:
                self.graph.parse(data=f.read(), format="xml")
            print("Parsed the NELL Graph...")
            print(f"No. points in graph: {len(self.graph)}")
            
            entity_relation_list = list(self.graph)
            graph_chunks = list(self.chunks(entity_relation_list, int(len(self.graph)/8)))
            vocab_chunks = list(self.chunks(list(self.ent_vocab), int(len(list(self.ent_vocab))/8)))
            threads = []
            datas = [None] * (len(graph_chunks)+len(vocab_chunks))
            ys = [None] * (len(graph_chunks)+len(vocab_chunks))

            i = 0
            for chunk in graph_chunks:
                t = threading.Thread(target=self.traverse_graph, args=(chunk, datas, ys, i))
                t.start()
                threads.append(t)
                i += 1

            for thread in threads:
                thread.join()

            threads = []
            for chunk in vocab_chunks:
                t = threading.Thread(target=self.traverse_vocab, args=(chunk, datas, ys, i))
                t.start()
                threads.append(t)
                i += 1

            for thread in threads:
                thread.join()

            data = list(itertools.chain.from_iterable(datas))
            y = list(itertools.chain.from_iterable(ys))

        else:
            print("Data file exists...")
            data = np.load(self.DATAX, allow_pickle=True)
            y = np.load(self.DATAY, allow_pickle=True)
        print("Created the training data...")
        return np.array(data), np.array(y)

    def pick_random(self, e1, r, e2):
        new_e = e1
        while (new_e, r, e2) in self.graph:
            new_e = random.sample(list(self.ent_vocab), 1)[0]

        return new_e

    def pick_random_2(self, e, e_list):
        new_e = e
        while new_e == e or f'{new_e}' == f'{e_list[0]}:{e_list[1]}':
            new_e = random.sample(list(self.ent_vocab), 1)[0]

        return new_e

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def traverse_graph(self, entity_relation_list, data, ys, i):
        x, y = [], []
        for e1, r, e2 in tqdm(entity_relation_list):
            if r.strip() != "concept:haswikipediaurl" and r.strip() != "concept:generalizations" and r.strip() != "concept:latitudelongitude":
                if f'{e1}' in list(self.ent_vocab) and f'{e2}' in list(self.ent_vocab) and f'{r}' in list(self.rel_vocab):
                    triple1 = np.array([self.ent_vocab[f'{e1}'], self.rel_vocab[f'{r}'], self.ent_vocab[f'{e2}']])
                    x.append(triple1)
                    y.append(1)
                    new_e1 = self.pick_random(e1, r, e2)
                    triple2 = np.array([self.ent_vocab[new_e1], self.rel_vocab[f'{r}'], self.ent_vocab[f'{e2}']])
                    x.append(triple2)
                    y.append(0)
        data[i] = x
        ys[i] = y

    def traverse_vocab(self, entity_list, data, ys, i):
        x, y = [], []
        # Generalizations
        for e in tqdm(entity_list):
            e_list = e.split(':')
            if e_list[0] == 'UNK':
                continue
            if f'{e_list[0]}:{e_list[1]}' in list(self.ent_vocab):
                triple1 = np.array([self.ent_vocab[f'{e}'], self.rel_vocab['concept:isa'], self.ent_vocab[f'{e_list[0]}:{e_list[1]}']])
                x.append(triple1)
                y.append(1)
                new_e = self.pick_random_2(e, e_list)
                triple2 = np.array([self.ent_vocab[new_e], self.rel_vocab['concept:isa'], self.ent_vocab[f'{e_list[0]}:{e_list[1]}']])
                x.append(triple2)
                y.append(0)
        data[i] = x
        ys[i] = y