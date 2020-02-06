from module.bertmodule import get_bert_config, get_bert_layer
from module.functionmodule import sigmoid_3
from tensorflow.keras import layers, Model, optimizers, metrics, losses
from bert import load_bert_weights

def create_model(max_seq_len):
    input_token = layers.Input(shape=(max_seq_len,), dtype='int32')
    bert_layer = get_bert_layer(get_bert_config("./pre-trained"))
    bert_representation = bert_layer(input_token)
    bert_lambda = layers.Lambda(lambda seq : seq[:, 0, :])(bert_representation)
    bert_dropout = layers.Dropout(0.25)(bert_lambda)
    dense1 = layers.Dense(64, activation='relu')(bert_dropout)
    dense1 = layers.Dropout(0.25)(dense1)
    output = layers.Dense(1, activation=sigmoid_3)(dense1)

    bertie = Model(inputs=input_token, outputs=output)
    bertie.build(input_shape=(None, max_seq_len))

    load_bert_weights(bert_layer, "./pre-trained/bert_model.ckpt")

    bertie.compile(optimizer=optimizers.Adam(),
                   loss=losses.MeanSquaredError(),
                   metrics=[metrics.RootMeanSquaredError()])
    bertie.summary()

    return bertie

if __name__ == '__main__':
    bertie = create_model(30)