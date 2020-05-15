import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, metrics, losses
import tensorflow_hub as hub
import os

from lib.models.layers.highway import Highway

def create_BERT_model():
    max_seq_length = 128
    sentence_in = layers.Input(shape=(), dtype=tf.string, name="sentence_in")
    embed = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2", trainable=False)(sentence_in)    # Expects a tf.string input tensor.
    # input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    # input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    # segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
    # albert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/1", trainable=False)
    # pooled_output, sequence_output = albert_layer([input_word_ids, input_mask, segment_ids])


    x = layers.Dense(64, activation='relu')(embed)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1)(x)

    bertie = Model(inputs=[sentence_in], outputs=x)

    bertie.compile(optimizer=optimizers.Adam(),
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    bertie.summary()

    return bertie