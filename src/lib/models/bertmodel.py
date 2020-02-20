from lib.models.module.bertmodule import get_bert_config, get_bert_layer, get_adapter_BERT_layer, freeze_bert_layers
from lib.models.module.functionmodule import sigmoid_3
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, metrics, losses
from bert import load_stock_weights
import os

def create_BERT_model(max_seq_len, adapter_size=64):
    pretrained_dir = f'{os.getcwd()}/lib/models/pre-trained'
    input_token = layers.Input(shape=(max_seq_len,), dtype='int32', name="token_input")
   
    # BERT things
    bert_layer = get_adapter_BERT_layer(pretrained_dir, adapter_size)
    bert_representation = bert_layer(input_token)
    bert_lambda = layers.Lambda(lambda seq : seq[:, 0, :])(bert_representation)
    bert_dropout = layers.Dropout(0.5)(bert_lambda)

    # Output things
    dense1 = layers.Dense(units=768, activation="tanh")(bert_dropout)
    dense1 = layers.Dropout(0.5)(dense1)
    output = layers.Dense(1, activation=sigmoid_3)(dense1)

    bertie = Model(inputs=input_token, outputs=output)

    bertie.build(input_shape=(None, max_seq_len))

    bertie.compile(optimizer=optimizers.Adam(clipnorm=1., clipvalue=0.5),
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    load_stock_weights(bert_layer, f'{pretrained_dir}/bert_model.ckpt')

    if adapter_size != None:
        freeze_bert_layers(bert_layer)

    bertie.summary()

    return bertie