import tensorflow as tf
from tensorflow.keras import layers


class MultiHeadedAttention(layers.Layer):
    def __init__(self, no_heads, embedding_size, dimension):
        super(MultiHeadedAttention, self).__init__()

        W_init = tf.random_normal_initializer()
        self.W_O = tf.Variable(initial_value=W_init(shape=(embedding_size, dimension),
                                                dtype='float32'),
                                trainable=True)