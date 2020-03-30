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

def create_HUMOR_model(feature_len : int, kb_len : int, kb_part : bool, word_encoder : bool, replaced : bool, replacement : bool) -> Model:
    """Create a humor model. 

    Feature length and KB length is for defining input sizes.
    KB part is for defining whether you want the knowledge base part or not.
    To select the word encoder you will have to set word_encoder. 
    Subsequently you will have to set replaced and replacement according to which inputs you want.
    If you don't at all want the word encoder set it to false.  
    
    Arguments:
        feature_len {int} -- The number of features
        kb_len {int} -- The length of the KB Vector
        kb_part {bool} -- Whether you want the KB part or not
        word_encoder {bool} -- Whether you want the word encoder or not.
        replaced {bool} -- If word encoder is inplace do you then want the replaced input.
        replacement {bool} -- If word encoder is inplace do you then want the replacement input.
    
    Returns:
        Model -- The compiled keras model.
    """
    ###### Feature Part
    input_features = layers.Input(shape=(feature_len,), dtype='float32', name="FeatureInput")

    feature_dense = layers.Dense(16, activation='relu', name="FeatureDense1")(input_features)
    feature_dense = layers.Dropout(0.5)(feature_dense)
    feature_dense = layers.Dense(16, activation='relu', name="FeatureDense2")(feature_dense)
    feature_dense = layers.Dropout(0.5)(feature_dense)

    outputs = [feature_dense]
    inputs = [input_features]
    ####################

    ###### Knowledge Part
    if kb_part:
        input_entities = layers.Input(shape=(kb_len,), dtype='int32', name="EntityInput")
        embeddings = np.load('../data/NELL/embeddings/entity.npy')
        entity_embedding = layers.Embedding(181544, 64, embeddings_initializer=initializers.Constant(embeddings), trainable=False, name="EntityEmbeddings")(input_entities)
        sum_layer = layers.Lambda(lambda x: backend.sum(x, axis=1, keepdims=False))(entity_embedding)
        entity_dense = layers.Dense(32, activation='relu', name="EntityDense1")(sum_layer)
        entity_dense = layers.Dropout(0.5)(entity_dense)
        entity_dense = layers.Dense(16, activation='relu', name="EntityDense2")(entity_dense)
        entity_dense = layers.Dropout(0.5)(entity_dense)
        outputs.append(entity_dense)
        inputs.append(input_entities)
    ####################

    ###### Sentence Part
    if word_encoder:
        
        sentence_in = layers.Input(shape=(), dtype=tf.string, name="sentence_in")
        embed = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1')(sentence_in)    # Expects a tf.string input tensor.
        sentence_dense = layers.Dense(64, activation='relu', name="SentenceDense1")(embed)
        sentence_dense = layers.Dropout(0.5)(sentence_dense)
        sentence_dense = layers.Dense(32, activation='relu', name="SentenceDense2")(sentence_dense)
        sentence_dense = layers.Dropout(0.5)(sentence_dense)
        sentence_dense = layers.Dense(16, activation='relu', name="SentenceDense3")(sentence_dense)
        sentence_dense = layers.Dropout(0.5)(sentence_dense)
        sentence_model = Model(sentence_in, sentence_dense, name="WordModel")

        concat_vector = []

        if replaced:
            input_replaced = layers.Input(shape=(), dtype=tf.string, name="ReplacedInput")
            concat_vector.append(sentence_model(input_replaced))
            inputs.append(input_replaced)
        if replacement:
            input_replacement = layers.Input(shape=(), dtype=tf.string, name="ReplacementInput")
            concat_vector.append(sentence_model(input_replacement))
            inputs.append(input_replacement)

        if replaced and replacement:
            concat_sentence = layers.Concatenate()(concat_vector)
            outputs.append(concat_sentence)
        else:
            outputs.append(concat_vector[0])
    #####################

    ###### Common Part
    if len(outputs) > 1:
        concat = layers.Concatenate()(outputs)
        output = layers.Dense(1, name="Output")(concat)
    else:
        output = layers.Dense(1, name="Output")(outputs[0])
    #  input_tokens, 
    HUMOR = Model(inputs=inputs, outputs=output, name="KBHumor")

    # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    opt = optimizers.Adam(lr=0.001)
    # opt = optimizers.Nadam(clipnorm=1., clipvalue=0.5)

    HUMOR.compile(optimizer=opt,
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    HUMOR.summary()

    return HUMOR