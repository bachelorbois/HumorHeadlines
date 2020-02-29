import json
import numpy as np
import nltk
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential, metrics
from tensorflow.keras.layers import Dense, Embedding, Flatten, Lambda, Conv2D
from tensorflow.keras.initializers import Constant
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

DATA_FILE = "../data/extra_data_sarcasm/Sarcasm_Headlines_Dataset.json"
DATA_FILE_v2 = "../data/extra_data_sarcasm/Sarcasm_Headlines_Dataset_v2.json"
EMBEDDING_DIM = 300

def avg_over_dimensions(x):
    x_avg = K.mean(x, axis=1, keepdims=False)
    return x_avg


class SarcasmClassifier():

    dataset = None
    vocab = None
    X_train, X_test, y_train, y_test = None, None, None, None
    max_langth = None
    embedding_matrix = None
    model = None

    @classmethod
    def run_preproc(cls):

        cls.merge_datasets(DATA_FILE, DATA_FILE_v2)
        cls.load_embeddings()

        proc_data, labels = cls.process_dataset()

        #sentences = process_sentences(tokenized_sentences, dictionary)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(proc_data, labels, test_size=0.2)

    @classmethod
    def process_dataset(cls):
        labels = []
        tokenized_sentences = []
        tokens_set = []
        cls.max_length = 0
        for row in cls.dataset:
            sentence = row['headline']
            tokens = nltk.word_tokenize(sentence)
            if len(tokens) < 27:
                labels.append(row['is_sarcastic'])
                tokenized_sentences.append(tokens)
                tokens_set.extend(tokens)
                if len(tokens) > cls.max_length:
                    cls.max_length = len(tokens)

        labels = np.array(labels)

        tokens_set = set(tokens_set)
        tokens_set.add('UNK')
        tok_to_id = {}
        id_to_tok = {}
        for i, tok in enumerate(tokens_set):
            tok_to_id[tok] = i+1
            id_to_tok[i+1] = tok

        with open('../data/sarcasm_utils/sarcasm_tokens_to_id.txt', 'w') as outfile:
            outfile.write(json.dumps(tok_to_id))
        
        with open('../data/sarcasm_utils/sarcasm_id_to_tokens.txt', 'w') as outfile:
            outfile.write(json.dumps(id_to_tok))

        proc_data = []
        for sentence in tokenized_sentences:
            proc_sent = []
            for token in sentence:
                proc_sent.append(tok_to_id[token])
            proc_data.append(proc_sent)

        proc_data = np.array(proc_data)

        proc_data = pad_sequences(proc_data, maxlen=cls.max_length, padding='post')

        cls.embedding_matrix = np.zeros((len(tokens_set)+1, EMBEDDING_DIM))
        for tok in tok_to_id.keys():
            i = tok_to_id[tok]
            try:
                cls.embedding_matrix[i] = cls.vocab[tok]
            except KeyError:
                cls.embedding_matrix[i] = cls.vocab['UNK']

        np.savetxt('../data/sarcasm_utils/sarcasm_embedding_matrix.txt', cls.embedding_matrix)

        return proc_data, labels

    @classmethod
    def run_model(cls):
        cls.model = Sequential()
        cls.model.add(Embedding(cls.embedding_matrix.shape[0],EMBEDDING_DIM,embeddings_initializer=Constant(cls.embedding_matrix),
                                input_length=cls.max_length,
                                trainable=False))
        cls.model.add(Lambda(avg_over_dimensions, output_shape=(EMBEDDING_DIM, )))
        cls.model.add(Dense(128, activation='relu'))
        cls.model.add(Dense(64, activation='relu'))
        cls.model.add(Dense(32, activation='relu'))
        cls.model.add(Dense(1, activation='sigmoid'))
        cls.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print(cls.model.summary())

        cls.model.fit(cls.X_train, cls.y_train, nb_epoch=15, validation_split=0.2)
        cls.model.evaluate(cls.X_test, cls.y_test)

        with open("lib/models/pre-trained/sarcasm_model_config.json", "w") as json_file:
            json_file.write(cls.model.to_json())

        cls.model.save("lib/models/pre-trained/sarcasm_model.h5")

    @classmethod
    def process_sentences(cls, tokenized_sentences, dictionary):
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

    @classmethod
    def load_embeddings(cls):
        dictionary = {}
        with open("../data/embeddings/wiki-news-300d-1M.vec") as infile:
            for line in infile:
                line = line.split()
                word = line[0]
                emb = np.array(line[1:], dtype='float')
                dictionary[word] = emb

        dictionary['UNK'] = np.array(list(dictionary.values())).mean()
        
        cls.vocab = dictionary

    @classmethod
    def read_file(cls, filename):
        data = []
        with open(filename) as f:
            for line in f:
                data.append(json.loads(line))

        return np.array(data)

    @classmethod
    def merge_datasets(cls, file1, file2):
        cls.dataset = np.concatenate((cls.read_file(file1), cls.read_file(file2)))

    @classmethod
    def process_sentence(cls, sentence):
        if not cls.vocab:
            with open('../data/sarcasm_utils/sarcasm_tokens_to_id.txt') as infile:
                tok_to_id = json.load(infile)
            
            with open('../data/sarcasm_utils/sarcasm_id_to_tokens.txt') as infile:
                id_to_tok = json.load(infile)

        proc_sentence = []
        for w in sentence:
            try:
                proc_sentence.append(tok_to_id[w])
            except KeyError:
                proc_sentence.append(tok_to_id['UNK'])
                    
        return np.array(proc_sentence)

    @classmethod
    def predict_sarcasm(cls, sentence):
        return cls.model.predict(sentence)
    

def run():
    classifier = SarcasmClassifier()
    classifier.run_preproc()
    classifier.run_model()

if __name__ == "__main__":
   run()