from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, initializers, optimizers, metrics, regularizers
import numpy as np

def create_MultiCNN_model():
    text_in = layers.Input(shape=(25,), dtype='int32', name="TextIn")

    glove = np.load('../data/embeddings/numpy/GloVe.npy', allow_pickle=True)
    glove_embed = layers.Embedding(input_dim=len(glove), output_dim=300, embeddings_initializer=initializers.Constant(glove), trainable=False, name="GloveEmbedding")(text_in)
    glove_conv = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(glove_embed)
    glove_drop = layers.Dropout(0.5)(glove_conv)
    glove_pool = layers.MaxPooling1D(pool_size=2)(glove_drop)
    glove_flat = layers.Flatten()(glove_pool)
    
    fasttext = np.load('../data/embeddings/numpy/fasttext.npy', allow_pickle=True)
    fasttext_embed = layers.Embedding(input_dim=len(fasttext), output_dim=300, embeddings_initializer=initializers.Constant(fasttext), trainable=False, name="FastTextEmbedding")(text_in)
    fasttext_conv = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(fasttext_embed)
    fasttext_drop = layers.Dropout(0.5)(fasttext_conv)
    fasttext_pool = layers.MaxPooling1D(pool_size=2)(fasttext_drop)
    fasttext_flat = layers.Flatten()(fasttext_pool)
    
    gnews = np.load('../data/embeddings/numpy/GNews.npy', allow_pickle=True)
    gnews_embed = layers.Embedding(input_dim=len(gnews), output_dim=300, embeddings_initializer=initializers.Constant(gnews), trainable=False, name="GNewsEmbedding")(text_in)
    gnews_conv = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(gnews_embed)
    gnews_drop = layers.Dropout(0.5)(gnews_conv)
    gnews_pool = layers.MaxPooling1D(pool_size=2)(gnews_drop)
    gnews_flat = layers.Flatten()(gnews_pool)
    
    custom = np.load('../data/embeddings/numpy/headline.npy', allow_pickle=True)
    custom_embed = layers.Embedding(input_dim=len(custom), output_dim=300, embeddings_initializer=initializers.Constant(custom), trainable=False, name="CustomEmbedding")(text_in)
    custom_conv = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(custom_embed)
    custom_drop = layers.Dropout(0.5)(custom_conv)
    custom_pool = layers.MaxPooling1D(pool_size=2)(custom_drop)
    custom_flat = layers.Flatten()(custom_pool)

    merged = layers.concatenate([glove_flat, fasttext_flat, gnews_flat, custom_flat])
    # interpretation
    x = layers.Dense(10, activation='relu')(merged)
    x = layers.Dense(1)(x)
    # dense = layers.Dense(64)(dense)
    # dense = layers.Dense(32)(dense)
    # dense = layers.Dense(16)(dense)
    # dense = layers.Dense(1)(dense)

    m = Model(text_in, x)
    m.compile(optimizer=optimizers.Adam(),
                   loss="mean_squared_error",
                   metrics=[metrics.RootMeanSquaredError()])

    m.summary()

    return m


