import json
import numpy as np
import nltk
from tensorflow.keras import layers, Sequential, metrics
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


def run_logistic_regression():
    DATA_FILE = "../data/extra_data_sarcasm/Sarcasm_Headlines_Dataset.json"
    DATA_FILE_v2 = "../data/extra_data_sarcasm/Sarcasm_Headlines_Dataset_v2.json"

    dataset = merge_datasets(DATA_FILE, DATA_FILE_v2)
    dictionary = load_embeddings()

    tokenized_sentences = []
    for row in dataset:
        sentence = row['headline']
        tokenized_sentences.append(nltk.word_tokenize(sentence))

    sentences = process_sentences(tokenized_sentences, dictionary)

    labels = []
    for line in dataset:
        labels.append(line['is_sarcastic'])
    labels = np.array(labels)


    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=15, validation_split=0.2)
    model.evaluate(X_test, y_test)

    with open("lib/models/pre-trained/sarcasm_model_config.json", "w") as json_file:
        json_file.write(model.to_json())

    model.save_weights("lib/models/pre-trained/sarcasm_model.h5")

    
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


def load_embeddings():
    dictionary = {}
    with open("../data/embeddings/wiki-news-300d-1M.vec") as infile:
        for line in infile:
            line = line.split()
            word = line[0]
            emb = np.array(line[1:], dtype='float')
            dictionary[word] = emb

    dictionary['UNK'] = np.array(list(dictionary.values())).mean()
    
    return dictionary

def read_file(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    return np.array(data)

def merge_datasets(file1, file2):
    return np.concatenate((read_file(file1), read_file(file2)))
    

if __name__ == "__main__":
    run_logistic_regression()