#!/usr/bin/python
from numpy.random import seed
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
seed(13377331)
from lib.training.humortraining import HumorTraining
from lib.training.namtraining import NAMTraining
from lib.models import ( create_HUMOR_model, 
                        create_NAM_model, 
                        create_HUMORX2_model, 
                        create_HUMOR2_model, 
                        create_BERT_model, 
                        create_CNN_model, 
                        create_KBLSTM_model,
                        create_MultiCNN_model )
from lib.models.tuning import HumorTuner, HumorTunerServer
from lib.inference import Task2Inference
from lib.parsing import read_task1_pb
from lib.features import PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature, NellKbFeature, AlbertTokenizer
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import load_model
from pyfiglet import Figlet
from PyInquirer import prompt
from kerastuner.tuners import Hyperband
import kerastuner
import json
import codecs
from collections import defaultdict
import plotly.graph_objects as go
import plotly
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from randomcolor import RandomColor
import gc

# mpl.rcParams['lines.linewidth'] = 5.5

def main(variant):
    HYPERBAND_MAX_EPOCHS = 50
    EXECUTION_PER_TRIAL = 2
    SEED = 13377331
    # ,'../data/FunLines/task-1/preproc/2_concat_train.bin'
    train_path, dev_path, test_path = ['../data/task-1/preproc/2_concat_train.bin'], ['../data/task-1/preproc/2_concat_dev.bin'], ['../data/task-1/preproc/2_concat_test.bin']
    if variant == 'HUMOR':
        params = json.load(open("./lib/models/tuning/model.json", 'r'))

        train_data = load_data(train_path)
        dev_data = load_data(dev_path)
        test_data = load_data(test_path)

        features = [PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature, NellKbFeature]
        
        train_data.AddFeatures(features)
        dev_data.AddFeatures(features)
        test_data.AddFeatures(features)

        features, train_y = train_data.GetFeatureVectors(), train_data.GetGrades()
        ins = {"FeatureInput": features[:,:4]}
        i = 4
        ins["EntityInput"] = features[:,i:]

        # text = np.load('../data/task-1/train_replaced.npy', allow_pickle=True) # 
        text = train_data.GetReplaced()
        ins["ReplacedInput"] = text

        # text = np.load('../data/task-1/train_edit.npy', allow_pickle=True) # 
        text = train_data.GetEdits()
        ins["ReplacementInput"] = text

        # Dev data
        dev_features, dev_y = dev_data.GetFeatureVectors(), dev_data.GetGrades()
        devIns = {"FeatureInput": dev_features[:,:4]}
        i = 4
        devIns["EntityInput"] = dev_features[:,i:]

        # text = np.load('../data/task-1/dev_replaced.npy', allow_pickle=True) # 
        text = dev_data.GetReplaced()
        devIns["ReplacedInput"] = text

        # text = np.load('../data/task-1/dev_edit.npy', allow_pickle=True) # 
        text = dev_data.GetEdits()
        devIns["ReplacementInput"] = text

        # Test data
        test_features, test_y = test_data.GetFeatureVectors(), test_data.GetGrades()
        testIns = {"FeatureInput": test_features[:,:4]}
        i = 4
        testIns["EntityInput"] = test_features[:,i:]

        # text = np.load('../data/task-1/test_replaced.npy', allow_pickle=True) # 
        text = test_data.GetReplaced()
        testIns["ReplacedInput"] = text

        # text = np.load('../data/task-1/test_edit.npy', allow_pickle=True) # 
        text = test_data.GetEdits()
        testIns["ReplacementInput"] = text

        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, mode='min', restore_best_weights=True)

        score = []
        for i in range(10):
            model = create_HUMOR2_model(4, 25, 128, params["hyperparameters"])
            model.fit(x=ins, y=train_y,
                    validation_data=(devIns, dev_y),
                    batch_size=16,
                    epochs=40,
                    shuffle=True,
                    callbacks=[early])
            
            preds = model.predict(testIns)
            score.append(mean_squared_error(test_y, round_numbers(preds), squared=False))
            print(score[i])
            del model
            # tf.reset_default_graph()
            tf.keras.backend.clear_session()
            gc.collect()
        score = np.array(score)
        print(f'{variant}: Mean: {score.mean()}, STD: {score.std()}')

    elif variant == 'HUMOR2':
        params = json.load(open("./lib/models/tuning/model2.json", 'r'))

        train_data = load_data(train_path)
        dev_data = load_data(dev_path)
        test_data = load_data(test_path)

        features = [PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature, NellKbFeature, AlbertTokenizer]
        
        train_data.AddFeatures(features)
        dev_data.AddFeatures(features)
        test_data.AddFeatures(features)

        features, train_y = train_data.GetFeatureVectors(), train_data.GetGrades()
        ins = {"FeatureInput": features[:,:4]}
        i = 4
        ins["EntityInput"] = features[:,i:i+25]
        i += 25
        ins["input_word_ids"] = features[:,i:i+128]
        i += 128
        ins["segment_ids"] = features[:,i:i+128]
        i += 128
        ins["input_mask"] = features[:,i:i+128]

        text = train_data.GetReplaced()
        ins["ReplacedInput"] = text

        text = train_data.GetEdits()
        ins["ReplacementInput"] = text

        # Dev data
        dev_features, dev_y = dev_data.GetFeatureVectors(), dev_data.GetGrades()
        devIns = {"FeatureInput": dev_features[:,:4]}
        i = 4
        devIns["EntityInput"] = dev_features[:,i:i+25]
        i += 25
        devIns["input_word_ids"] = dev_features[:,i:i+128]
        i += 128
        devIns["segment_ids"] = dev_features[:,i:i+128]
        i += 128
        devIns["input_mask"] = dev_features[:,i:i+128]

        text = dev_data.GetReplaced()
        devIns["ReplacedInput"] = text

        text = dev_data.GetEdits()
        devIns["ReplacementInput"] = text

        # Test data
        test_features, test_y = test_data.GetFeatureVectors(), test_data.GetGrades()
        testIns = {"FeatureInput": test_features[:,:4]}
        i = 4
        testIns["EntityInput"] = test_features[:,i:i+25]
        i += 25
        testIns["input_word_ids"] = test_features[:,i:i+128]
        i += 128
        testIns["segment_ids"] = test_features[:,i:i+128]
        i += 128
        testIns["input_mask"] = test_features[:,i:i+128]

        text = test_data.GetReplaced()
        testIns["ReplacedInput"] = text

        text = test_data.GetEdits()
        testIns["ReplacementInput"] = text

        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, mode='min', restore_best_weights=True)

        score = []
        for i in range(10):
            model = create_HUMOR2_model(4, 25, 128, params["hyperparameters"])
            model.fit(x=ins, y=train_y,
                    validation_data=(devIns, dev_y),
                    batch_size=16,
                    epochs=25,
                    shuffle=True,
                    callbacks=[early])
            
            preds = model.predict(testIns)
            score.append(mean_squared_error(test_y, preds, squared=False))
            del model
            # tf.reset_default_graph()
            tf.keras.backend.clear_session()
            gc.collect()
        score = np.array(score)
        print(f'{variant}: Mean: {score.mean()}, STD: {score.std()}')
    elif variant == 'TASK2INFER':
        model = './headline_regression/20200308-194029-BEST/weights/final.hdf5'
        infer = Task2Inference(model, '../data/task-2/preproc/2_concat_test.bin')
        infer.predict('../data/task-2/predictions/task-2-output.csv')
    elif variant == 'TESTINFER':
        preds = 'task-1-output.context.csv'
        test = load_data(test_path)
        y = test.GetGrades()
        with open(preds, 'r') as f:
            i = 0
            pred_list = []
            for line in f:
                if i == 0:
                    i = 1
                else:
                    pred_list.append(float(line.strip().split(',')[1]))
        rmse = mean_squared_error(y, np.array(pred_list), squared=False)
        print(rmse)

    elif variant == 'NAM':
        model = create_NAM_model(1, 181544, 832)
        data_path = '../data/NELL/NELLRDF.xml'
        ent_vocab = '../data/NELL/NELLWordNetVocab.txt'
        rel_vocab = '../data/NELL/NELLRelVocab.txt'
        trainer = NAMTraining(model, data_path, ent_vocab, rel_vocab)
        trainer.train(30, 2048)
        trainer.test()
    elif variant == 'TUNING':
        model = HumorTuner(4, 20)
        tuner = Hyperband(model, 
                        max_epochs=HYPERBAND_MAX_EPOCHS,
                        objective=kerastuner.Objective("val_root_mean_squared_error", direction="min"),
                        seed=SEED,
                        executions_per_trial=EXECUTION_PER_TRIAL,
                        hyperband_iterations=2,
                        directory=f'tuning_hyperband',
                        project_name='ContextHumor'
                    )

        tuner.search_space_summary()

        ## Loading the data
        train_data = load_data(train_path)
        dev_data = load_data(dev_path)
        features = [PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature, NellKbFeature]
        train_data.AddFeatures(features)
        dev_data.AddFeatures(features)

        features, train_y = train_data.GetFeatureVectors(), train_data.GetGrades()
        ins = {"FeatureInput": features[:,:4]}
        i = 4
        ins["EntityInput"] = features[:,i:i+20]

        ins["ReplacedInput"] = train_data.GetReplaced()
        ins["ReplacementInput"] = train_data.GetEdits()

        # Dev data
        dev_features, dev_y = dev_data.GetFeatureVectors(), dev_data.GetGrades()
        devIns = {"FeatureInput": dev_features[:,:4]}
        i = 4
        devIns["EntityInput"] = dev_features[:,i:i+20]

        devIns["ReplacedInput"] = dev_data.GetReplaced()
        devIns["ReplacementInput"] = dev_data.GetEdits()

        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0005, patience=2, mode='min', restore_best_weights=True)

        tuner.oracle.hyperband_iterations = 2

        tuner.search(ins, train_y, 
                    epochs=HYPERBAND_MAX_EPOCHS, 
                    batch_size=64, 
                    validation_data=(devIns, dev_y),
                    callbacks=[early])

        tuner.results_summary()
    elif variant == 'TUNINGSERVER': 
        params = json.load(open("./lib/models/tuning/model.json", 'r'))
        model = HumorTunerServer(4, 20, 128, params["hyperparameters"])
        tuner = Hyperband(model, 
                        max_epochs=HYPERBAND_MAX_EPOCHS,
                        objective=kerastuner.Objective("val_root_mean_squared_error", direction="min"),
                        seed=SEED,
                        executions_per_trial=EXECUTION_PER_TRIAL,
                        hyperband_iterations=1,
                        directory=f'tuning_hyperband',
                        project_name='ContextHumor'
                    )

        tuner.search_space_summary()

        ## Loading the data
        train_data = load_data(train_path)
        dev_data = load_data(dev_path)
        features = [PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature, NellKbFeature, AlbertTokenizer]
        train_data.AddFeatures(features)
        dev_data.AddFeatures(features)

        features, train_y = train_data.GetFeatureVectors(), train_data.GetGrades()
        ins = {"FeatureInput": features[:,:4]}
        i = 4
        ins["EntityInput"] = features[:,i:i+20]
        i += 20
        ins["input_word_ids"] = features[:,i:i+128]
        i += 128
        ins["segment_ids"] = features[:,i:i+128]
        i += 128
        ins["input_mask"] = features[:,i:i+128]

        text = train_data.GetReplaced()
        ins["ReplacedInput"] = text

        text = train_data.GetEdits()
        ins["ReplacementInput"] = text

        # Dev data
        dev_features, dev_y = dev_data.GetFeatureVectors(), dev_data.GetGrades()
        devIns = {"FeatureInput": dev_features[:,:4]}
        i = 4
        devIns["EntityInput"] = dev_features[:,i:i+20]
        i += 20
        devIns["input_word_ids"] = dev_features[:,i:i+128]
        i += 128
        devIns["segment_ids"] = dev_features[:,i:i+128]
        i += 128
        devIns["input_mask"] = dev_features[:,i:i+128]

        text = dev_data.GetReplaced()
        devIns["ReplacedInput"] = text

        text = dev_data.GetEdits()
        devIns["ReplacementInput"] = text

        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0005, patience=2, mode='min', restore_best_weights=True)

        tuner.search(ins, train_y, 
                    epochs=HYPERBAND_MAX_EPOCHS, 
                    batch_size=64, 
                    validation_data=(devIns, dev_y),
                    callbacks=[early])

        tuner.results_summary()
    elif variant == 'PLOT':
        axes = ["feature_units1", "feature_units2","entity_units1","entity_units2","sentence_units1","sentence_units2","sentence_units3"]
        models = json.load(open("./lib/models/tuning/result_summary.json", 'r'))
        params = defaultdict(list)

        for model in models["top_10"]:
            t_id = model["TrialID"]
            model_param = json.load(open(f"./tuning_hyperband/HumorHumor/trial_{t_id}/trial.json", "r"))
            for a in axes:
                params[a].append(model_param["hyperparameters"]["values"][a])
            params["score"].append(model["Score"])

        fig = go.Figure(data=
            go.Parcoords(
                line_color='green',
                dimensions=list([
                    dict(range=[8, 128],
                        label='Feature Layer 1',
                        values=params[axes[0]]),
                    dict(range=[8, 128],
                        label='Feature Layer 2',
                        values=params[axes[1]]),
                    dict(range=[8, 128],
                        label='Knowledge Layer 1',
                        values=params[axes[2]]),
                    dict(range=[8, 128],
                        label='Knowledge Layer 2',
                        values=params[axes[3]]),
                    dict(range=[32, 512],
                        label='Word Layer 2',
                        values=params[axes[4]]),
                    dict(range=[32, 512],
                        label='Word Layer 1',
                        values=params[axes[5]]),
                    dict(range=[8, 128],
                        label='Word Layer 2',
                        values=params[axes[6]]),
                    dict(range=[0, 1],
                        label='Root Mean Square Error',
                        values=params["score"]),
                ])
            )
        )

        fig.show()

    elif variant == 'MultiCNN':
        train_data = load_data(train_path)
        dev_data = load_data(dev_path)
        test_data = load_data(test_path)
        with codecs.open('../data/vocab/train_vocab.json', encoding='utf-8') as fp:
            vocab_dict = json.load(fp)

        max_length = longest(train_data.GetTokenizedWEdit())
        ins = {"TextIn": convert_to_index(vocab_dict, train_data.GetTokenizedWEdit(), max_length)}
        train_y = train_data.GetGrades()
        devIns = {"TextIn": convert_to_index(vocab_dict, dev_data.GetTokenizedWEdit(), max_length)}
        dev_y = dev_data.GetGrades()
        testIns = {"TextIn": convert_to_index(vocab_dict, test_data.GetTokenizedWEdit(), max_length)}
        test_y = test_data.GetGrades()
        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, mode='min', restore_best_weights=True)
        lr_schedule = create_learning_rate_scheduler(max_learn_rate=1e-1,
                                                    end_learn_rate=1e-6,
                                                    warmup_epoch_count=15,
                                                    total_epoch_count=40)
        score = []
        for i in range(10):
            model = create_MultiCNN_model()
            model.fit(x=ins, y=train_y,
                    validation_data=(devIns, dev_y),
                    batch_size=16,
                    epochs=40,
                    shuffle=True,
                    callbacks=[early])
            
            preds = model.predict(testIns)
            score.append(mean_squared_error(test_y, preds, squared=False))
            del model
            # tf.reset_default_graph()
            tf.keras.backend.clear_session()
            gc.collect()
        score = np.array(score)
        print(f'{variant}: Mean: {score.mean()}, STD: {score.std()}')

        # preds = model.predict(devIns)
        # ids = dev_data.GetIDs()
        # out = np.stack((ids, preds.flatten()), axis=-1)
        # Save the predictions to file
        # np.savetxt(f'../plots/{variant}.csv', out, header='id,pred', fmt="%d,%1.8f")

        # print(f'Mean of preds: {preds.mean()}, STD of preds: {preds.std()}, Mean of true: {dev_y.mean()}, STD of true: {dev_y.std()}')
        # bins = np.linspace(0, 3, 50)
        # plt.hist(preds, bins=bins, alpha=0.5, label="preds")
        # plt.hist(dev_y, bins=bins, alpha=0.5, label="true")
        # plt.legend(loc='upper right')
        # plt.show()
        # del model

    elif variant == 'CNN':
        # model = create_CNN_model()
        train_data = load_data(train_path)
        dev_data = load_data(dev_path)
        test_data = load_data(test_path)
        with codecs.open('../data/vocab/train_vocab.json', encoding='utf-8') as fp:
            vocab_dict = json.load(fp)

        max_length = longest(train_data.GetTokenizedWEdit())
        ins = {"TextIn": convert_to_index(vocab_dict, train_data.GetTokenizedWEdit(), max_length)}
        train_y = train_data.GetGrades()
        devIns = {"TextIn": convert_to_index(vocab_dict, dev_data.GetTokenizedWEdit(), max_length)}
        dev_y = dev_data.GetGrades()
        testIns = {"TextIn": convert_to_index(vocab_dict, test_data.GetTokenizedWEdit(), max_length)}
        test_y = test_data.GetGrades()
        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, mode='min', restore_best_weights=True)
        score = []
        for i in range(10):
            model = create_CNN_model()
            model.fit(x=ins, y=train_y,
                    validation_data=(devIns, dev_y),
                    batch_size=16,
                    epochs=40,
                    shuffle=True,
                    callbacks=[early])
            
            preds = model.predict(testIns)
            score.append(mean_squared_error(test_y, preds, squared=False))
            del model
            # tf.reset_default_graph()
            tf.keras.backend.clear_session()
            gc.collect()
        score = np.array(score)
        print(f'{variant}: Mean: {score.mean()}, STD: {score.std()}')

        # preds = model.predict(devIns)
        # ids = dev_data.GetIDs()
        # out = np.stack((ids, preds.flatten()), axis=-1)
        # Save the predictions to file
        # np.savetxt(f'../plots/{variant}.csv', out, header='id,pred', fmt="%d,%1.8f")

        # print(f'Mean of preds: {preds.mean()}, STD of preds: {preds.std()}, Mean of true: {dev_y.mean()}, STD of true: {dev_y.std()}')
        # bins = np.linspace(0, 3, 50)
        # plt.hist(preds, bins=bins, alpha=0.5, label="preds")
        # plt.hist(dev_y, bins=bins, alpha=0.5, label="true")
        # plt.legend(loc='upper right')
        # plt.show()
        # del model

    elif variant == 'KBLSTM':
        train_data = load_data(train_path)
        train_data.AddFeatures([NellKbFeature])
        dev_data = load_data(dev_path)
        dev_data.AddFeatures([NellKbFeature])
        test_data = load_data(test_path)
        test_data.AddFeatures([NellKbFeature])
        with codecs.open('../data/vocab/train_vocab.json', encoding='utf-8') as fp:
            vocab_dict = json.load(fp)

        max_length = longest(train_data.GetTokenizedWEdit())
        train = convert_to_index(vocab_dict, train_data.GetTokenizedWEdit(), max_length)
        ins = {"TextIn": train, "EntityInput": train_data.GetFeatureVectors()}
        train_y = train_data.GetGrades()
        dev = convert_to_index(vocab_dict, dev_data.GetTokenizedWEdit(), max_length)
        devIns = {"TextIn": dev, "EntityInput": dev_data.GetFeatureVectors()}
        dev_y = dev_data.GetGrades()
        test = convert_to_index(vocab_dict, test_data.GetTokenizedWEdit(), max_length)
        testIns = {"TextIn": test, "EntityInput": test_data.GetFeatureVectors()}
        test_y = test_data.GetGrades()
        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, mode='min', restore_best_weights=True)
        score = []
        for i in range(10):
            model = create_KBLSTM_model()
            model.fit(x=ins, y=train_y,
                    validation_data=(devIns, dev_y),
                    batch_size=16,
                    epochs=40,
                    shuffle=True,
                    callbacks=[early])
            
            preds = model.predict(testIns)
            score.append(mean_squared_error(test_y, preds, squared=False))
            del model
            # tf.reset_default_graph()
            tf.keras.backend.clear_session()
            gc.collect()
        score = np.array(score)
        print(f'{variant}: Mean: {score.mean()}, STD: {score.std()}')

        # preds = model.predict(devIns)
        # ids = dev_data.GetIDs()
        # out = np.stack((ids, preds.flatten()), axis=-1)
        # Save the predictions to file
        # np.savetxt(f'../plots/{variant}.csv', out, header='id,pred', fmt="%d,%1.8f")

        # print(f'Mean of preds: {preds.mean()}, STD of preds: {preds.std()}, Mean of true: {dev_y.mean()}, STD of true: {dev_y.std()}')
        # bins = np.linspace(0, 3, 50)
        # plt.hist(preds, bins=bins, alpha=0.5, label="preds")
        # plt.hist(dev_y, bins=bins, alpha=0.5, label="true")
        # plt.legend(loc='upper right')
        # plt.show()
        del model

    elif variant == 'NNLM':
        train_data = load_data(train_path)
        dev_data = load_data(dev_path)
        test_data = load_data(test_path)

        ins = {"sentence_in": train_data.GetEditSentences()}
        devIns = {"sentence_in": dev_data.GetEditSentences()}
        testIns = {"sentence_in": test_data.GetEditSentences()}
        train_y = train_data.GetGrades()
        dev_y = dev_data.GetGrades()
        test_y = test_data.GetGrades()
        early = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, mode='min', restore_best_weights=True)
        score = []
        for i in range(10):
            model = create_BERT_model()
            model.fit(x=ins, y=train_y,
                    validation_data=(devIns, dev_y),
                    batch_size=16,
                    epochs=40,
                    shuffle=True,
                    callbacks=[early])
            
            preds = model.predict(testIns)
            score.append(mean_squared_error(test_y, preds, squared=False))
            del model
            # tf.reset_default_graph()
            tf.keras.backend.clear_session()
            gc.collect()
        score = np.array(score)
        print(f'{variant}: Mean: {score.mean()}, STD: {score.std()}')

        # preds = model.predict(devIns)
        # ids = dev_data.GetIDs()
        # out = np.stack((ids, preds.flatten()), axis=-1)
        # Save the predictions to file
        # np.savetxt(f'../plots/{variant}.csv', out, header='id,pred', fmt="%d,%1.8f")

        # print(f'Mean of preds: {preds.mean()}, STD of preds: {preds.std()}, Mean of true: {dev_y.mean()}, STD of true: {dev_y.std()}')
        # bins = np.linspace(0, 3, 50)
        # plt.hist(preds, bins=bins, alpha=0.5, label="preds")
        # plt.hist(dev_y, bins=bins, alpha=0.5, label="true")
        # plt.legend(loc='upper right')
        # plt.show()
        # del model

    elif variant == 'LINEAR':
        train = load_data(train_path)
        dev = load_data(dev_path)

        features = [PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature]
        
        train.AddFeatures(features)
        dev.AddFeatures(features)

        X, y = train.GetFeatureVectors(), train.GetGrades()
        X_dev, dev_y = dev.GetFeatureVectors(), dev.GetGrades()
        
        reg = LinearRegression(n_jobs=-1).fit(X, y)
        
        preds = reg.predict(X_dev)
        rmse = mean_squared_error(test_y, preds, squared=False)
        # ids = dev.GetIDs()
        # out = np.stack((ids, preds.flatten()), axis=-1)
        # Save the predictions to file
        # np.savetxt(f'../plots/{variant}.csv', out, header='id,pred', fmt="%d,%1.8f")

        # print(f'Mean of preds: {preds.mean()}, STD of preds: {preds.std()}, Mean of true: {dev_y.mean()}, STD of true: {dev_y.std()}')
    elif variant == 'VOCAB':
        embed_path = '../data/embeddings/numpy/headline.npy'
        print("Loading embeddings and vocab...")
        model = Word2Vec.load('../data/embeddings/headlineEmbeds.bin')
        print("Loaded embeddings...")
        with codecs.open('../data/vocab/train_vocab.funlines.json', encoding='utf-8') as fp:
            vocab_dict = json.load(fp)
        print("Loaded vocab...")
        
        embed_matrix = np.zeros((len(vocab_dict), 300))
        i = 0
        for k, v in vocab_dict.items():
            try:
                embed_matrix[v] = model.wv.get_vector(k)
            except KeyError:
                # print(f'{k} does not exist in FastText embeddings')
                i += 1
        print(len(vocab_dict), i)
        print("Created the embedding matrix...")
        np.save(embed_path, embed_matrix)
        print("Saved the new embeddings...")
    elif variant == 'WORD2VEC':
        print("Loading data...")
        headline_paths = ['../data/extra_data_sarcasm/Sarcasm_Headlines_Dataset_v2.json']
        headlines = []
        for headline_path in headline_paths:
            with open(headline_path, 'r') as fp:
                for line in fp:
                    d = json.loads(line)
                    headlines.append(text_to_word_sequence(d["headline"]))

        train_data = load_data(train_path)
        print("Train model...")
        print(len(headlines))
        headlines.extend(train_data.GetTokenizedWEdit())
        print(len(headlines))
        model = Word2Vec(headlines, size=300, window=14, workers=4, min_count=1)

        vocab = list(model.wv.vocab)
        print(len(vocab))

        print("Saving model...")
        model.save('../data/embeddings/headlineEmbeds.bin')
    elif variant == 'PCA':
        model = PCA(n_components=3)
        entities = np.load('../data/NELL/embeddings/entity.npy')
        labels = load_vocab('../data/NELL/NELLWordNetVocab_proc.txt')
        top_100 = {}
        with open('../data/NELL/top_100_nell.txt', 'r') as f:
            for line in f:
                label = line.strip()
                top_100[label] = entities[labels[label]]

        # print(entities[:4])
        # print(labels[:4])

        pca_ent = model.fit_transform(list(top_100.values()))

        # create_dendrogram(list(top_100.values()), list(top_100.keys()), 'ward')


        # print(pca_ent.shape)
        # print(pca_ent[:10])
        rand_color = RandomColor()
        fig = go.Figure(data=[go.Scatter3d(
            x=pca_ent[:,0],
            y=pca_ent[:,1],
            z=pca_ent[:,2],
            mode='markers',
            text=list(top_100.keys()),
            marker=dict(
                size=12,
                color=rand_color.generate(count=100),
                colorscale='Viridis',
                opacity=0.8
            )
        )])

        plotly.offline.plot(fig, filename="NELLPCA.html")
    elif variant == 'MEAN':
        files = ['CNN.csv', 'context.csv', 'KBLSTM.csv', 'LINEAR.csv', 'MultiCNN.csv', 'NNLM.csv', 'simple.csv']
        for f in files:
            with open(f'../plots/{f}', 'r') as fp:
                i = 0
                vals = []
                for line in fp:
                    if i == 0:
                        i += 1
                        continue
                    vals.append(float(line.strip().split(',')[1]))
            vals = np.array(vals)
            mean, std = vals.mean(), vals.std()
            print(f'{f.split(".")[0]}: Mean: {mean}, STD: {std}')
    elif variant == 'COEF':
        train = load_data(train_path)

        features = [PhoneticFeature, PositionFeature, DistanceFeature, SentLenFeature]
        
        train.AddFeatures(features)

        X, y = train.GetFeatureVectors(), train.GetGrades()
        y = np.reshape(y, (-1, 1))
        print(y.shape)
        z = np.concatenate((X, y), axis=-1).T
        
        coef = np.corrcoef(z).round(decimals=4)

        np.savetxt("coef.csv", coef, delimiter=',')
    elif variant == 'ALBERT':
        model = create_BERT_model()
        train = load_data(train_path)
        dev = load_data(dev_path)
        test = load_data(test_path)

        features = [AlbertTokenizer]

        train.AddFeatures(features)
        dev.AddFeatures(features)
        test.AddFeatures(features)

        features, indexes = dev.GetFeatureVectors(), dev.GetIndexes()
        
        ins = {}
        i=0
        ins["input_word_ids"] = features[:,i:i+128]
        i += 128
        ins["segment_ids"] = features[:,i:i+128]
        i += 128
        ins["input_mask"] = features[:,i:i+128]

        preds = model.predict(ins)
        words_train = []
        for i, pred in enumerate(preds):
            words_train.append(pred[indexes[i]])
        words_train = np.array(words_train)
        print(words_train.shape)

        np.save("./dev_edit.npy", words_train)
    elif variant == 'MEDIAN':
        train_data = load_data(train_path)
        test_data = load_data(test_path)

        train_y = train_data.GetGrades()
        test_y = test_data.GetGrades()

        pred = np.mean(train_y)
        print("Median", pred)

        pred_y = np.array([pred] * len(test_y))
        rmse = mean_squared_error(test_y, pred_y, squared=False)
        print("RMSE", rmse)

def convert_to_index(vocab, sents, max_length=27):
    # print(max_length)
    seq = np.zeros((len(sents), max_length))
    for i, sent in enumerate(sents):
        for j, word in enumerate(sent):
            try:
                seq[i, j] = vocab[word]
            except KeyError:
                pass

    return seq

def create_dendrogram(X, labels, link):
    linked = linkage(X, link)

    plt.figure(figsize=(25, 15))
    plt.yticks(fontsize=25)
    dendrogram(linked,
                orientation='top',
                labels=labels,
                distance_sort='descending',
                show_leaf_counts=True,
                leaf_rotation=-90,
                leaf_font_size=10)
    plt.show()

def load_vocab(path):
    with open(path, 'r') as f:
        lines = {'UNK': 0}
        i = 1
        for line in f:
            lines[line.strip().split(':')[-1]] = i
            i += 1

    return lines

def round_numbers(numbers):
    labels = np.arange(0.0, 3.2, 0.2)
    new_nums = []
    for number in numbers:
        d = {l:abs(l-number) for l in labels}
        new_nums.append(min(d, key=d.get))
    return np.array(new_nums)

def load_data(paths):
    hc = None
    for path in paths:
        with open(path, 'rb') as fd:
            if not hc:
                hc = read_task1_pb(fd)
            else: 
                hc.extend(read_task1_pb(fd))
    return hc

def longest(l):
    # if(not isinstance(l, list)): return(0)
    return(max([len(subl) for subl in l if isinstance(subl, list)]))

def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                        end_learn_rate=1e-7,
                                        warmup_epoch_count=10,
                                        total_epoch_count=90):
    import math
    def lr_scheduler_exp_decay(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)

    def lr_scheduler_step_decay(epoch):
        initial_lrate = 0.005
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler_step_decay, verbose=1)

    return learning_rate_scheduler


if __name__ == '__main__':
    f = Figlet(font='slant')
    print(f.renderText('Humor Regression'))
    q = [
        {
            'type': 'list',
            'name': 'variant',
            'message': 'What type of training/testing varint do you want?',
            'choices': ['HUMOR', 'HUMOR2', 'TASK2INFER', 'TESTINFER', 'NAM', 'TUNING', 'TUNINGSERVER', 'MultiCNN', 'CNN', 'KBLSTM', 'NNLM', 'VOCAB', 'WORD2VEC', 'PCA', 'MEAN', 'COEF', 'ALBERT', 'MEDIAN']
        }
    ]
    var = prompt(q)['variant']
    main(variant=var)
    # main(variant='MultiCNN')
    # main(variant='KBLSTM')
    # main(variant='NNLM')
    # main(variant='LINEAR')
    # main(variant='MEDIAN')
