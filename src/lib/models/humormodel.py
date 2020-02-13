import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, metrics

from lib.models.module.functionmodule import sigmoid_3

def create_HUMOR_model(phonetics_len : int, distance_len : int, position_len : int, cluster_len : int, length_len : int) -> Model:
    input_phonetics = layers.Input(shape=(phonetics_len,), dtype='float32', name="phonetics_input")
    input_distance = layers.Input(shape=(distance_len,), dtype='float32', name="distance_input")
    input_position = layers.Input(shape=(position_len,), dtype='float32', name="position_input")
    input_cluster = layers.Input(shape=(cluster_len,), dtype='float32', name="cluster_input")
    input_length = layers.Input(shape=(length_len,), dtype='float32', name="length_input")

    concat = layers.Concatenate()([input_phonetics, input_distance, input_position, input_cluster, input_length])

    dense1 = layers.Dense(32, activation='relu')(concat)
    dense1_dropout = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(16, activation='relu')(dense1_dropout)
    dense2_dropout = layers.Dropout(0.5)(dense2)
    output = layers.Dense(1, activation=sigmoid_3)(dense2_dropout)

    HUMOR = Model(inputs=[input_phonetics, input_distance, input_position, input_cluster, input_length], outputs=[output])

    HUMOR.compile(optimizer=optimizers.Adam(clipnorm=1., clipvalue=0.5),
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    HUMOR.summary()

    return HUMOR