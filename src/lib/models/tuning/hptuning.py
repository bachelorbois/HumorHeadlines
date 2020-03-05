from kerastuner import HyperModel
from tensorflow.keras import layers, Model, optimizers, metrics
import tensorflow_hub as hub
import tensorflow as tf

class HumorTuner(HyperModel):
    def __init__(self, feature_len):
        self.feature_len = feature_len
        self.nnlm_path = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1'

    def build(self, hp):
        ###### Feature Part
        input_features = layers.Input(shape=(self.feature_len,), dtype='float32', name="feature_input")

        feature_dense = layers.Dense(units=hp.Int(
                                        'feature_units1',
                                        min_value=8,
                                        max_value=128,
                                        step=16,
                                        default=24
                                    ), activation=hp.Choice(
                                        'feature_dense_activation1',
                                        values=['relu', 'tanh', 'sigmoid'],
                                        default='relu'
                                    ),  name="FeatureDense1")(input_features)
        feature_dense = layers.Dropout(rate=hp.Float(
                                                'feature_dropout_1',
                                                min_value=0.0,
                                                max_value=0.5,
                                                default=0.25,
                                                step=0.05,
                                            ))(feature_dense)
        feature_dense = layers.Dense(units=hp.Int(
                                        'feature_units2',
                                        min_value=8,
                                        max_value=128,
                                        step=16,
                                        default=24
                                    ), activation=hp.Choice(
                                        'feature_dense_activation2',
                                        values=['relu', 'tanh', 'sigmoid'],
                                        default='relu'
                                    ), name="FeatureDense2")(feature_dense)
        feature_dense = layers.Dropout(rate=hp.Float(
                                                'feature_dropout_2',
                                                min_value=0.0,
                                                max_value=0.5,
                                                default=0.25,
                                                step=0.05,
                                            ))(feature_dense)
        ####################

        ###### Sentence Part
        input_replaced = layers.Input(shape=(), dtype=tf.string, name="replaced_input")
        input_replacement = layers.Input(shape=(), dtype=tf.string, name="repacement_input")
        
        sentence_in = layers.Input(shape=(), dtype=tf.string, name="sentence_in")
        embed = hub.KerasLayer(self.nnlm_path)(sentence_in)    # Expects a tf.string input tensor.
        sentence_dense = layers.Dense(units=hp.Int(
                                        'sentence_units1',
                                        min_value=32,
                                        max_value=512,
                                        step=32,
                                        default=64
                                    ), activation=hp.Choice(
                                        'sentence_dense_activation1',
                                        values=['relu', 'tanh', 'sigmoid'],
                                        default='relu'
                                    ), name="SentenceDense1")(embed)
        sentence_dense = layers.Dropout(rate=hp.Float(
                                                'sentence_dropout_1',
                                                min_value=0.0,
                                                max_value=0.5,
                                                default=0.25,
                                                step=0.05,
                                            ))(sentence_dense)
        sentence_dense = layers.Dense(units=hp.Int(
                                        'sentence_units2',
                                        min_value=32,
                                        max_value=512,
                                        step=16,
                                        default=32
                                    ), activation=hp.Choice(
                                        'sentence_dense_activation2',
                                        values=['relu', 'tanh', 'sigmoid'],
                                        default='relu'
                                    ), name="SentenceDense2")(sentence_dense)
        sentence_dense = layers.Dropout(rate=hp.Float(
                                                'sentence_dropout_2',
                                                min_value=0.0,
                                                max_value=0.5,
                                                default=0.25,
                                                step=0.05,
                                            ))(sentence_dense)
        sentence_dense = layers.Dense(units=hp.Int(
                                        'sentence_units3',
                                        min_value=8,
                                        max_value=128,
                                        step=8,
                                        default=128
                                    ), activation=hp.Choice(
                                        'sentence_dense_activation3',
                                        values=['relu', 'tanh', 'sigmoid'],
                                        default='relu'
                                    ), name="SentenceDense3")(sentence_dense)
        sentence_dense = layers.Dropout(rate=hp.Float(
                                                'sentence_dropout_3',
                                                min_value=0.0,
                                                max_value=0.5,
                                                default=0.25,
                                                step=0.05,
                                            ))(sentence_dense)

        sentence_model = Model(sentence_in, sentence_dense, name="SentenceModel")

        concat_sentence = layers.Concatenate()([sentence_model(input_replaced), sentence_model(input_replacement)])
        #####################

        ###### Common Part
        concat = layers.Concatenate()([feature_dense, concat_sentence])
        # output = layers.Dense(16, activation='relu', name="OutoutDense1")(feature_dense)
        output = layers.Dense(1, name="Output")(concat)
        #  input_tokens, 
        HUMOR = Model(inputs=[input_features, input_replaced, input_replacement], outputs=output)

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