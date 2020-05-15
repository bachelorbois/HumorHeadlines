import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import lib

with open("../data/task-1/preproc/2_concat_train.bin", "rb") as fd:
    train_data = lib.read_task1_pb(fd)

with open("../data/task-1/preproc/2_concat_dev.bin", "rb") as fd:
    dev_data = lib.read_task1_pb(fd)

feats = [
    lib.features.DistanceFeature,
    lib.features.PositionFeature,
    lib.features.SentLenFeature,
    lib.features.PhoneticFeature]

for i, feat in enumerate(feats):

    train_data.ClearFeatures()
    dev_data.ClearFeatures()

    train_data.AddFeature(feat)
    dev_data.AddFeature(feat)

    X = train_data.GetFeatureVectors()
    y = train_data.GetGrades()


    # Write to files
    with open(f"plot_{i}.csv", "w") as fd:
        writer = csv.writer(fd, delimiter=",", quotechar='"')

        for i in range(len(y)):
            writer.writerow([*X[i], y[i]])


    X_dev = dev_data.GetFeatureVectors()
    y_true = dev_data.GetGrades()

    reg = LinearRegression().fit(X, y)

    y_pred = reg.predict(X_dev)

    print(i, X[0],
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )
