import tensorflow as tf
from tensorflow.keras import layers

class ScaledAttention(layers.Layer):
    def __init__(self, embedding_size, dimension):
        super(ScaledAttention, self).__init__()
        self.dim = dimension
        W_init = tf.random_normal_initializer()
        self.W_q = tf.Variable(initial_value=W_init(shape=(embedding_size, dimension),
                                                dtype='float32'),
                                trainable=True)
        self.W_k = tf.Variable(initial_value=W_init(shape=(embedding_size, dimension),
                                                dtype='float32'),
                                trainable=True)
        self.W_v = tf.Variable(initial_value=W_init(shape=(embedding_size, dimension),
                                                dtype='float32'),
                                trainable=True)

    def call(self, inputs):
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)
        soft = tf.softmax(tf.divide(tf.matmul(Q, tf.transpose(K)), tf.sqrt(self.dim)))
        return tf.matmul(soft, V)