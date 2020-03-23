from tensorflow.keras import layers, Model, models, optimizers
import tensorflow as tf
import tensorflow_hub as hub

def create_HUMORX2_model(feature_len : int, kb_len : int, **kwargs) -> Model:
    ### Headline 1
    input_features_hl1 =    layers.Input(shape=(feature_len,), dtype='float32', name="FeatureInputHL1")
    input_entities_hl1 =    layers.Input(shape=(kb_len,), dtype='int32', name="EntityInputHL1")
    input_replaced_hl1 =    layers.Input(shape=(), dtype=tf.string, name="ReplacedInputHL1")
    input_replacement_hl1 = layers.Input(shape=(), dtype=tf.string, name="ReplacementInputHL1")

    ### Headline 2
    input_features_hl2 =    layers.Input(shape=(feature_len,), dtype='float32', name="FeatureInputHL2")
    input_entities_hl2 =    layers.Input(shape=(kb_len,), dtype='int32', name="EntityInputHL2")
    input_replaced_hl2 =    layers.Input(shape=(), dtype=tf.string, name="ReplacedInputHL2")
    input_replacement_hl2 = layers.Input(shape=(), dtype=tf.string, name="ReplacementInputHL2")

    task_1 = models.load_model("./headline_regression/20200308-194029-BEST/weights/final.hdf5", custom_objects={'KerasLayer': hub.KerasLayer})

    HL1 = task_1([input_features_hl1, input_entities_hl1, input_replaced_hl1, input_replacement_hl1])
    HL2 = task_1([input_features_hl2, input_entities_hl2, input_replaced_hl2, input_replacement_hl2])

    concat = layers.Concatenate()([HL1, HL2])

    output = layers.Dense(1, activation="sigmoid", name="Task2Output")(concat)

    TASK2 = Model(inputs=[  input_features_hl1, input_entities_hl1, input_replaced_hl1, input_replacement_hl1, 
                            input_features_hl2, input_entities_hl2, input_replaced_hl2, input_replacement_hl2],
                outputs=output, name="Task2Humor")

    opt = optimizers.Adam(lr=0.005)

    TASK2.compile(optimizer=opt, 
                loss="binary_crossentropy",
                metrics=["accuracy"])

    TASK2.summary()

    return TASK2
