import numpy as np
import pandas as pd
from collections import defaultdict
import re

import sys
import os
import codecs
import tqdm
import json

from tensorflow.keras import layers, Model, initializers, optimizers, metrics

from lib.models.layers.attention import AttentionWeightedAverage
from lib.models.layers.knowledge import KnowledgeLayer

def create_KBLSTM_model():
    text_in = layers.Input(shape=(25,), dtype='int32', name="TextIn")
    input_entities = layers.Input(shape=(25,), dtype='int32', name="EntityInput")
    embed_path = "../data/embeddings/numpy/GNews.npy"
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

    embed_layer = layers.Embedding(input_dim=len(embed_matrix), output_dim=300, trainable=False, embeddings_initializer=initializers.Constant(embed_matrix))(text_in)

    embeddings = np.load('../data/NELL/embeddings/entity.npy')
    entity_embedding = layers.Embedding(181544, 100, embeddings_initializer=initializers.Constant(embeddings), trainable=False, name="EntityEmbeddings")(input_entities)

    HIDDEN_LAYER_DIMENSION = 64

    state_vector = layers.Bidirectional(layers.LSTM(HIDDEN_LAYER_DIMENSION, dropout=0.5, return_sequences=True))(embed_layer)

    attention_layer = AttentionWeightedAverage()(state_vector)

    attention_layer = layers.Dense(100, activation='relu')(attention_layer)

    hidden = KnowledgeLayer()([attention_layer,entity_embedding])

    # attention_layer = layers.Dense(64, activation='relu')(attention_layer)

    hidden = layers.add([hidden, attention_layer])

    preds = layers.Dense(1)(hidden)

    m = Model([text_in,input_entities], preds)
    m.compile(optimizer=optimizers.Adam(),
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    m.summary()

    return m