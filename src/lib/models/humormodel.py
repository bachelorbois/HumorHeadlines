import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, metrics, models
import tensorflow_hub as hub
import os

from lib.models.module.functionmodule import sigmoid_3
from lib.models.layers.elmo import ElmoEmbeddingLayer

def create_HUMOR_model(feature_len : int, token_len : int) -> Model:
    ###### Feature Part
    input_features = layers.Input(shape=(feature_len,), dtype='float32', name="feature_input")
    # input_tokens = layers.Input(shape=(token_len,), dtype='int32', name="token_input")

    # sarcasm = models.load_model("lib/models/pre-trained/sarcasm_model.h5")
    # sarcasm.trainable = False

    # concat = layers.Concatenate()([input_features, sarcasm(input_tokens)])

    feature_dense = layers.Dense(16, activation='relu', name="FeatureDense1")(input_features)
    feature_dense = layers.Dropout(0.50)(feature_dense)
    feature_dense = layers.Dense(16, activation='relu', name="FeatureDense2")(feature_dense)
    feature_dense = layers.Dropout(0.50)(feature_dense)
    ####################

    ###### Sentence Part
    input_replaced = layers.Input(shape=(), dtype=tf.string, name="replaced_input")
    input_replacement = layers.Input(shape=(), dtype=tf.string, name="repacement_input")
    
    sentence_in = layers.Input(shape=(), dtype=tf.string, name="sentence_in")
    embed = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1')(sentence_in)    # Expects a tf.string input tensor.
    sentence_dense = layers.Dense(128, activation='relu', name="SentenceDense1")(embed)
    sentence_dense = layers.Dropout(0.50)(sentence_dense)
    sentence_dense = layers.Dense(64, activation='relu', name="SentenceDense2")(sentence_dense)
    sentence_dense = layers.Dropout(0.50)(sentence_dense)
    sentence_dense = layers.Dense(32, activation='relu', name="SentenceDense3")(sentence_dense)
    sentence_dense = layers.Dropout(0.50)(sentence_dense)

    sentence_model = Model(sentence_in, sentence_dense)

    concat_sentence = layers.Concatenate()([sentence_model(input_replaced), sentence_model(input_replacement)])
    #####################

    ###### Common Part
    concat = layers.Concatenate()([feature_dense, concat_sentence])
    output = layers.Dense(16, activation='relu', name="OutoutDense1")(concat)
    output = layers.Dense(1, name="Outout")(output)
    #  input_tokens,
    HUMOR = Model(inputs=[input_features, input_replaced, input_replacement], outputs=output)

    # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    opt = optimizers.Adam(lr=0.01)
    # opt = optimizers.Nadam(clipnorm=1., clipvalue=0.5)

    HUMOR.compile(optimizer=opt,
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    HUMOR.summary()

    return HUMOR