import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, metrics, initializers, regularizers
import tqdm
import numpy as np
import codecs
import json
import os

from lib.models.layers.highway import Highway

def create_CNN_model():
    text_in = layers.Input(shape=(25,), dtype='int32', name="TextIn")
    embed_path = "../data/embeddings/numpy/GloVe.npy"
    print("Loading embeddings...")
    if not os.path.isfile(embed_path):
        embeddings = {}
        with codecs.open('../data/embeddings/wiki-news-300d-1m.vec', encoding='utf-8') as f:
            for line in tqdm.tqdm(f):
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
        
        with codecs.open('../data/vocab/train_vocab.funlines.json', encoding='utf-8') as fp:
            vocab_dict = json.load(fp)
            
        embed_matrix = np.zeros((len(vocab_dict), 300))
        i = 0
        for k, v in vocab_dict.items():
            try:
                embed_matrix[v] = embeddings[k]
            except KeyError:
                # print(f'{k} does not exist in FastText embeddings')
                i += 1
        print(len(vocab_dict), i)
        np.save(embed_path, embed_matrix)
    else:
        embed_matrix = np.load(embed_path, allow_pickle=True)

    embed_layer = layers.Embedding(input_dim=len(embed_matrix), output_dim=300, embeddings_initializer=initializers.Constant(embed_matrix), trainable=False)(text_in)

    x = layers.Conv1D(100, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(embed_layer)
    x = layers.Dropout(0.5)(x)
    # x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(100, 6, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(100, 7, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64)(x)
    # x = layers.Dropout(0.5)(x)
    # x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    # x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    # x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(32)(x)
    # x = layers.Dropout(0.5)(x)
    # x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    # x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    # x = Highway()(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(1)(x)

    m = Model(text_in, x)
    m.compile(optimizer=optimizers.Adam(),
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    m.summary()

    return m