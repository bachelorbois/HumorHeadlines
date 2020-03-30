from kerastuner import HyperModel
from tensorflow.keras import layers, Model, optimizers, metrics, initializers, backend
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

class HumorTuner(HyperModel):
    def __init__(self, feature_len, kb_len):
        self.feature_len = feature_len
        self.kb_len = kb_len
        self.nnlm_path = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1'

    def build(self, hp):
        ###### Feature Part
        input_features = layers.Input(shape=(self.feature_len,), dtype='float32', name="feature_input")
        input_entities = layers.Input(shape=(self.kb_len,), dtype='int32', name="entity_input")

        feature_dense = layers.Dense(units=hp.Int(
                                        'feature_units1',
                                        min_value=8,
                                        max_value=128,
                                        step=16,
                                        default=24
                                    ), activation='relu',  name="FeatureDense1")(input_features)
        feature_dense = layers.Dropout(rate=0.5)(feature_dense)
        feature_dense = layers.Dense(units=hp.Int(
                                        'feature_units2',
                                        min_value=8,
                                        max_value=128,
                                        step=16,
                                        default=24
                                    ), activation='relu', name="FeatureDense2")(feature_dense)
        feature_dense = layers.Dropout(rate=0.5)(feature_dense)

        embeddings = np.load('../data/NELL/embeddings/entity.npy')
        entity_embedding = layers.Embedding(181544, 64, embeddings_initializer=initializers.Constant(embeddings), trainable=False, name="EntityEmbeddings")(input_entities)
        sum_layer = layers.Lambda(lambda x: backend.sum(x, axis=1, keepdims=False))(entity_embedding)
        entity_dense = layers.Dense(units=hp.Int(
                                        'entity_units1',
                                        min_value=8,
                                        max_value=128,
                                        step=16,
                                        default=24
                                    ), activation='relu', name="EntityDense1")(sum_layer)
        entity_dense = layers.Dropout(rate=0.5)(entity_dense)
        entity_dense = layers.Dense(units=hp.Int(
                                        'entity_units2',
                                        min_value=8,
                                        max_value=128,
                                        step=16,
                                        default=24
                                    ), activation='relu', name="EntityDense2")(entity_dense)
        entity_dense = layers.Dropout(rate=0.5)(entity_dense)
        ####################

        ###### Sentence Part
        input_replaced = layers.Input(shape=(), dtype=tf.string, name="replaced_input")
        input_replacement = layers.Input(shape=(), dtype=tf.string, name="replacement_input")
        
        sentence_in = layers.Input(shape=(), dtype=tf.string, name="sentence_in")
        embed = hub.KerasLayer(self.nnlm_path)(sentence_in)    # Expects a tf.string input tensor.
        sentence_dense = layers.Dense(units=hp.Int(
                                        'sentence_units1',
                                        min_value=32,
                                        max_value=512,
                                        step=32,
                                        default=64
                                    ), activation='relu', name="SentenceDense1")(embed)
        sentence_dense = layers.Dropout(rate=0.5)(sentence_dense)
        sentence_dense = layers.Dense(units=hp.Int(
                                        'sentence_units2',
                                        min_value=32,
                                        max_value=512,
                                        step=16,
                                        default=32
                                    ), activation='relu', name="SentenceDense2")(sentence_dense)
        sentence_dense = layers.Dropout(rate=0.5)(sentence_dense)
        sentence_dense = layers.Dense(units=hp.Int(
                                        'sentence_units3',
                                        min_value=8,
                                        max_value=128,
                                        step=8,
                                        default=128
                                    ), activation='relu', name="SentenceDense3")(sentence_dense)
        sentence_dense = layers.Dropout(rate=0.5)(sentence_dense)

        sentence_model = Model(sentence_in, sentence_dense, name="SentenceModel")

        concat_sentence = layers.Concatenate()([sentence_model(input_replaced), sentence_model(input_replacement)])
        #####################

        ###### Common Part
        concat = layers.Concatenate()([feature_dense, concat_sentence, entity_dense])
        # output = layers.Dense(16, activation='relu', name="OutoutDense1")(feature_dense)
        output = layers.Dense(1, name="Output")(concat)
        #  input_tokens, 
        HUMOR = Model(inputs=[input_features, input_replaced, input_replacement, input_entities], outputs=output)

        # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = optimizers.Adam(lr=hp.Float(
                                    'learning_rate',
                                    min_value=1e-5,
                                    max_value=1e-1,
                                    sampling='LOG',
                                    default=1e-2
                                ))
        # opt = optimizers.Nadam(clipnorm=1., clipvalue=0.5)

        HUMOR.compile(optimizer=opt,
                    loss="mean_squared_error",
                    metrics=[metrics.RootMeanSquaredError()])

        HUMOR.summary()

        return HUMOR