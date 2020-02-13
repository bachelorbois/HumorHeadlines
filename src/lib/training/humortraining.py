from tensorflow.keras import Model

from lib.models import create_HUMOR_model
from lib.parsing.parser import read_task1_pb
from lib.features import PhoneticFeature, Positionfeature, DistanceFeature, ClusterFeatures

class HumorTraining:
    def __init__(self, Humor : Model, train_path : str, test_path : str):
        humor = Humor

        self.train_data = self.load_data(train_path)
        self.test_data = self.load_data(test_path)

        features = [PhoneticFeature, Positionfeature, DistanceFeature, ClusterFeatures]

        self.train_data.AddFeatures(features)

    def train(self):
        features = self.train_data.GetFeatureVectors()

        print(features)

    def predict(self):
        pass
    
    @staticmethod
    def load_data(path):
        with open(path, 'rb') as fd:
            data = read_task1_pb(fd)

        return data