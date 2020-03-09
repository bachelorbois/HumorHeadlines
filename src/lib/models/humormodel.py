import tensorflow as tf
from tensorflow.keras import (Model,
                            layers, 
                            optimizers, 
                            metrics, 
                            models, 
                            initializers, 
                            backend)
import tensorflow_hub as hub
import os
import numpy as np

from lib.models.module.functionmodule import sigmoid_3
from lib.models.layers.elmo import ElmoEmbeddingLayer

def create_HUMOR_model(feature_len : int, kb_len) -> Model:
    ###### Feature Part
    input_features = layers.Input(shape=(feature_len,), dtype='float32', name="FeatureInput")
    input_entities = layers.Input(shape=(kb_len,), dtype='int32', name="EntityInput")

    feature_dense = layers.Dense(16, activation='relu', name="FeatureDense1")(input_features)
    feature_dense = layers.Dropout(0.50)(feature_dense)
    feature_dense = layers.Dense(16, activation='relu', name="FeatureDense2")(feature_dense)
    feature_dense = layers.Dropout(0.50)(feature_dense)

    embeddings = np.load('../data/NELL/embeddings/entity.npy')
    entity_embedding = layers.Embedding(181544, 64, embeddings_initializer=initializers.Constant(embeddings), trainable=False, name="EntityEmbeddings")(input_entities)
    sum_layer = layers.Lambda(lambda x: backend.sum(x, axis=1, keepdims=False))(entity_embedding)
    entity_dense = layers.Dense(32, activation='relu', name="EntityDense1")(sum_layer)
    entity_dense = layers.Dropout(0.50)(entity_dense)
    entity_dense = layers.Dense(16, activation='relu', name="EntityDense2")(entity_dense)
    entity_dense = layers.Dropout(0.50)(entity_dense)
    ####################

    ###### Sentence Part
    input_replaced = layers.Input(shape=(), dtype=tf.string, name="ReplacedInput")
    input_replacement = layers.Input(shape=(), dtype=tf.string, name="ReplacementInput")
    
    sentence_in = layers.Input(shape=(), dtype=tf.string, name="sentence_in")
    embed = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1')(sentence_in)    # Expects a tf.string input tensor.
    sentence_dense = layers.Dense(64, activation='relu', name="SentenceDense1")(embed)
    sentence_dense = layers.Dropout(0.50)(sentence_dense)
    sentence_dense = layers.Dense(32, activation='relu', name="SentenceDense2")(sentence_dense)
    sentence_dense = layers.Dropout(0.50)(sentence_dense)
    sentence_dense = layers.Dense(16, activation='relu', name="SentenceDense3")(sentence_dense)
    sentence_dense = layers.Dropout(0.50)(sentence_dense)

    sentence_model = Model(sentence_in, sentence_dense, name="SentenceModel")

    concat_sentence = layers.Concatenate()([sentence_model(input_replaced), sentence_model(input_replacement)])
    #####################

    ###### Common Part
    concat = layers.Concatenate()([feature_dense, concat_sentence, entity_dense])
    # output = layers.Dense(16, activation='relu', name="OutputDense1")(concat)
    # output = layers.Dropout(0.50)(output)
    output = layers.Dense(1, name="Output")(concat)
    #  input_tokens, 
    HUMOR = Model(inputs=[input_features, input_entities, input_replaced, input_replacement], outputs=output, name="HumanHumor")

    # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    opt = optimizers.Adam(lr=0.005)
    # opt = optimizers.Nadam(clipnorm=1., clipvalue=0.5)

    HUMOR.compile(optimizer=opt,
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    HUMOR.summary()

    return HUMOR