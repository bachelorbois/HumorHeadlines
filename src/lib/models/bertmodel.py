from lib.models.module.bertmodule import get_bert_config, get_bert_layer
from lib.models.module.functionmodule import sigmoid_3
from tensorflow.keras import layers, Model, optimizers, metrics, losses
from bert import load_stock_weights
import os

def create_model(max_seq_len):
    pretrained_dir = f'{os.getcwd()}/lib/models/pre-trained'
    input_token = layers.Input(shape=(max_seq_len,), dtype='int64')
   
    # BERT things
    bert_layer = get_bert_layer(get_bert_config(pretrained_dir))
    bert_representation = bert_layer(input_token)
    bert_lambda = layers.Lambda(lambda seq : seq[:, 0, :])(bert_representation)
    bert_dropout = layers.Dropout(0.25)(bert_lambda)

    # Output things
    dense1 = layers.Dense(64, activation='relu')(bert_dropout)
    dense1 = layers.Dropout(0.25)(dense1)
    output = layers.Dense(1, activation=sigmoid_3)(dense1)

    bertie = Model(inputs=input_token, outputs=output)

    bertie.build(input_shape=(None, max_seq_len))

    bertie.compile(optimizer=optimizers.RMSprop(0.001),
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    load_stock_weights(bert_layer, f'{pretrained_dir}/bert_model.ckpt')

    bertie.summary()

    return bertie

if __name__ == '__main__':
    bertie = create_model(30)