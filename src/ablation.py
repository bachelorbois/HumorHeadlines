import numpy as np
import tensorflow as tf

np.random.seed(13377331)
tf.random.set_seed(13377331)

from tensorflow.keras.callbacks import CSVLogger

import lib

features = [
    # lib.features.ClusterFeature,
    lib.features.DistanceFeature,
    lib.features.PhoneticFeature,
    lib.features.PositionFeature,
    lib.features.SentLenFeature
]

with open("../data/task-1/preproc/2_concat_train.bin", "rb") as fd:
    data = lib.read_task1_pb(fd)

with open("../data/task-1/preproc/2_concat_dev.bin", "rb") as fd:
    data_dev = lib.read_task1_pb(fd)


def train(l, i):
    data.ClearFeatures()
    data.AddFeatures(l)
    feat_len = len(data[0].GetFeatureVector())

    featureV, y_train = data.GetFeatureVectors(), data.GetGrades()
    ins = {"feature_input": featureV}

    ins["replaced_input"] = data.GetReplaced()

    ins["repacement_input"] = data.GetEdits()

    # Dev
    data_dev.ClearFeatures()
    data_dev.AddFeatures(l)

    dev_features, y_dev = data_dev.GetFeatureVectors(), data_dev.GetGrades()
    dev_ins = {"feature_input": dev_features}

    dev_ins["replaced_input"] = data_dev.GetReplaced()

    dev_ins["repacement_input"] = data_dev.GetEdits()


    humor = lib.models.create_HUMOR_model(feat_len)

    logger = CSVLogger(
        f"test-{i}.csv", separator=',', append=False
    )

    humor.fit(x=ins, y=y_train,
                validation_data=(dev_ins, y_dev),
                batch_size=1024,
                epochs=20,
                shuffle=True,
                callbacks=[logger])

    del humor

for i in range(len(features)):
    print(f"Leaving out {features[i]}!")
    l = features[:i] + features[i+1:]
    train(l, i)

train(features, "baseline")
