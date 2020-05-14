from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

import numpy as np

class KnowledgeLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('uniform')
        super(KnowledgeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(100, 100),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        #ents = np.load('../data/NELL/embeddings/entity.npy')[:10000]
        #self.concept = self.add_weight(shape=ents.shape,
        #                                name='{}_concept'.format(self.name),
        #                                initializer=initializers.Constant(ents),
        #                                trainable=False)
        # self.trainable_weights = [self.W]
        super(KnowledgeLayer, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, inputs, **kwargs):
        layer_output = inputs[0]
        concept = inputs[1]
        weight_layer = K.dot(layer_output,self.W)
        a = K.repeat_elements(weight_layer, K.int_shape(concept)[1],axis=1)
        shape = K.int_shape(weight_layer)
        b = K.reshape(a, (-1, K.int_shape(concept)[1], shape[1]))
        alpha = K.sum(b * concept, axis=2)
        att_weights = alpha / (K.sum(alpha, axis=1, keepdims=True))
        att_weights = K.expand_dims(att_weights,-1)
        knowledge_output = K.sum(att_weights * concept, axis=1)
        return knowledge_output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],512)