import numpy as np
import lib
from lib.parsing import Headline
from lib.features import Feature
import csv
from sklearn.metrics import mean_squared_error
from math import sqrt
from lib.models.module.functionmodule import RMSE


class Baseline():

    DATA_FILE = "../data/task-1/preproc/1_original_train.bin"
    length = []
    map_length_score = {}
    sentences = []
    actual = []

    @classmethod
    def run(cls):
        with open(cls.DATA_FILE, "rb") as fd:
            headlinecollection = lib.read_task1_pb(fd)

            for hl in headlinecollection:
                cls.length.append(len(hl.sentence))
                cls.sentences.append(hl.sentence)
                if not hl.avg_grade:
                    cls.actual.append(0)
                else:
                    cls.actual.append(hl.avg_grade)
            
        unit = 3/(max(cls.length)-min(cls.length))

        sum = -unit
        for i in range(min(cls.length),max(cls.length)+1):
            sum = sum+unit
            cls.map_length_score[i] = sum

        pred = []
        for l in cls.length:
            pred.append(cls.map_length_score[l])

        
        rmse = RMSE(cls.actual, pred)

        with open("../data/task-1/preproc/1_original_dev.bin", "rb") as fd:
            headlinecollection = lib.read_task1_pb(fd)

            pred_on_dev = []
            for hl in headlinecollection:
                idx = hl.id
                try:
                    pred_on_dev.append([idx, cls.map_length_score[len(hl.sentence)]])
                except KeyError:
                    if len(hl.sentence>max(cls.length)):
                        pred_on_dev.append([idx, 3.0])
                    else:
                        if len(hl.sentence<min(cls.length)):
                            pred_on_dev.append([idx, 0.0])
                

        with open("../predictions/task-1-output.csv", 'w', newline='') as outfile:
            wr = csv.writer(outfile)
            wr.writerow(['id','pred'])
            wr.writerows(pred_on_dev)

        


