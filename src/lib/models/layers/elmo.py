import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


class ElmoEmbeddingLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
        self.dimensions = 1024
        self.trainable = False

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=self.trainable)

        self.trainable_weights += tf.trainable_variables(scope="^elmo_module/.*")
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(tf.squeeze(tf.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)