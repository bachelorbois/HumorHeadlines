import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, metrics
import tensorflow_hub as hub

from lib.models.module.functionmodule import sigmoid_3
from lib.models.layers.elmo import ElmoEmbeddingLayer

def create_HUMOR_model(feature_len : int, embeds : bool = False) -> Model:
    input_features = layers.Input(shape=(feature_len,), dtype='float32', name="feature_input")
    
    if (embeds):
        input_text = layers.Input(shape=(), dtype=tf.string, name="string_input")
        embed = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1')(input_text)    # Expects a tf.string input tensor.
        concat = layers.Concatenate()([input_features, embed])
        dense1 = layers.Dense(128, activation='relu')(concat)
    else:
        dense1 = layers.Dense(128, activation='relu')(input_features)
    
    dense1_dropout = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(64, activation='relu')(dense1_dropout)
    dense2_dropout = layers.Dropout(0.5)(dense2)
    dense3 = layers.Dense(32, activation='relu')(dense2_dropout)
    dense3_dropout = layers.Dropout(0.5)(dense3)
    output = layers.Dense(1, activation=sigmoid_3)(dense3_dropout)

    if (embeds):
        HUMOR = Model(inputs=[input_features, input_text], outputs=output)
    else:
        HUMOR = Model(inputs=input_features, outputs=output)

    HUMOR.compile(optimizer=optimizers.Adam(clipnorm=1., clipvalue=0.5),
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    HUMOR.summary()

    return HUMOR