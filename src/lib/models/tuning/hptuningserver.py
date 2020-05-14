from kerastuner import HyperModel
from tensorflow.keras import layers, Model, optimizers, metrics, initializers, backend
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

class HumorTunerServer(HyperModel):
    def __init__(self, feature_len, kb_len, max_seq_length, parameters):
        self.feature_len = feature_len
        self.kb_len = kb_len
        self.nnlm_path = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1'
        self.albert_path = "https://tfhub.dev/tensorflow/albert_en_base/1"
        self.parameters = parameters
        self.max_seq_length = max_seq_length

    def build(self, hp):
        ###### Feature Part
        input_features = layers.Input(shape=(self.feature_len,), dtype='float32', name="FeatureInput")
        input_entities = layers.Input(shape=(self.kb_len,), dtype='int32', name="EntityInput")

        feature_dense = layers.Dense(self.parameters["feature_units1"], activation=self.parameters["feature_dense_activation1"], name="FeatureDense1")(input_features)
        feature_dense = layers.Dropout(self.parameters["feature_dropout_1"])(feature_dense)
        feature_dense = layers.Dense(self.parameters["feature_units2"], activation=self.parameters["feature_dense_activation2"], name="FeatureDense2")(feature_dense)
        feature_dense = layers.Dropout(self.parameters["feature_dropout_2"])(feature_dense)

        embeddings = np.load('../data/NELL/embeddings/entity.npy')
        entity_embedding = layers.Embedding(181544, 64, embeddings_initializer=initializers.Constant(embeddings), trainable=False, name="EntityEmbeddings")(input_entities)
        sum_layer = layers.Lambda(lambda x: backend.sum(x, axis=1, keepdims=False))(entity_embedding)
        entity_dense = layers.Dense(self.parameters["entity_units1"], activation=self.parameters["entity_dense_activation1"], name="EntityDense1")(sum_layer)
        entity_dense = layers.Dropout(self.parameters["entity_dropout_1"])(entity_dense)
        entity_dense = layers.Dense(self.parameters["entity_units2"], activation=self.parameters["entity_dense_activation2"], name="EntityDense2")(entity_dense)
        entity_dense = layers.Dropout(self.parameters["entity_dropout_2"])(entity_dense)
        ####################

        ###### Sentence Part
        input_replaced = layers.Input(shape=(), dtype=tf.string, name="ReplacedInput")
        input_replacement = layers.Input(shape=(), dtype=tf.string, name="ReplacementInput")

        sentence_in = layers.Input(shape=(), dtype=tf.string, name="sentence_in")
        word_embed = hub.KerasLayer(self.nnlm_path)(sentence_in)
        sentence_dense = layers.Dense(self.parameters["sentence_units1"], activation=self.parameters["sentence_dense_activation1"], name="SentenceDense1")(word_embed)
        sentence_dense = layers.Dropout(self.parameters["sentence_dropout_1"])(sentence_dense)
        sentence_dense = layers.Dense(self.parameters["sentence_units2"], activation=self.parameters["sentence_dense_activation2"], name="SentenceDense2")(sentence_dense)
        sentence_dense = layers.Dropout(self.parameters["sentence_dropout_2"])(sentence_dense)
        sentence_dense = layers.Dense(self.parameters["sentence_units3"], activation=self.parameters["sentence_dense_activation3"], name="SentenceDense3")(sentence_dense)
        sentence_dense = layers.Dropout(self.parameters["sentence_dropout_3"])(sentence_dense)

        sentence_model = Model(sentence_in, sentence_dense, name="WordModel")

        concat_sentence = layers.Concatenate()([sentence_model(input_replaced), sentence_model(input_replacement)])
        #####################

        ###### Albert
        input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="segment_ids")
        albert_layer = hub.KerasLayer(self.albert_path,
                                    trainable=False)
        pooled_output, sequence_output = albert_layer([input_word_ids, input_mask, segment_ids])

        context_dense = layers.Dense(hp.Int('contextUnits1',
                                        min_value=32,
                                        max_value=512,
                                        step=32), activation=hp.Choice('activation1', ['relu', 'tanh', 'sigmoid']), name="ContextDense1")(pooled_output)
        context_dense = layers.Dropout(0.5)(context_dense)
        context_dense = layers.Dense(hp.Int('contextUnits2',
                                        min_value=32,
                                        max_value=512,
                                        step=32), activation=hp.Choice('activation2', ['relu', 'tanh', 'sigmoid']), name="ContextDense2")(context_dense)
        context_dense = layers.Dropout(0.5)(context_dense)

        ###### Common Part
        concat = layers.Concatenate()([feature_dense, concat_sentence, entity_dense, context_dense])
        # output = layers.Dense(16, activation='relu', name="OutputDense1")(concat)
        # output = layers.Dropout(0.50)(output)
        output = layers.Dense(1, name="Output")(concat)
        #  input_tokens, 
        HUMOR = Model(inputs=[input_features, input_entities, input_replaced, input_replacement, input_word_ids, input_mask, segment_ids], outputs=output, name="KBHumor")

        # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = optimizers.Adam(lr=self.parameters["learning_rate"])
        # opt = optimizers.Nadam(clipnorm=1., clipvalue=0.5)

        HUMOR.compile(optimizer=opt,
                    loss="mean_squared_error",
                    metrics=[metrics.RootMeanSquaredError()])

        HUMOR.summary()

        return HUMOR