from itertools import count
import numpy as np
import sys
from numpy.lib.twodim_base import tri
import matplotlib.pyplot as plt
import argparse
import os
import math
import csv
from pathlib import Path
import random
import math
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def configure_svm(kernel):

    model = svm.SVC(100, random_state=42, kernel=kernel)

    return model

def train(data, labels, model):
        
    model_trained = model.fit(data, labels)

    return model_trained



def predict(data, model):
        
    list_prediction = []

    for feat in data:

        feat = feat.reshape([1, -1]) if(len(feat.shape) == 1) else feat

        score = model.predict(feat)

        list_prediction.append(score)        

    return list_prediction


parser = argparse.ArgumentParser(description='Gender Prediction with KNN using cross-database',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

parser.add_argument('--female', '-f', 
                    type=str, 
                    help='input of the features corresponding to the female group')

parser.add_argument('--male', '-m', 
                    type=str, 
                    help='input of the features corresponding to the male group')

parser.add_argument('--train', '-t', 
                    type=str, 
                    help='input of the features corresponding to the train group')

parser.add_argument('--testing', '-p', 
                    type=str, 
                    help='input of the features corresponding to the test group')

parser.add_argument('--database', '-cdb', 
                    type=str, 
                    help='input to the dir where are the features')

parser.add_argument('--iter', '-i', 
                    type=int, 
                    help='number of iterations')

parser.add_argument('--output', '-o', 
                    type=str, 
                    help='output')

args = parser.parse_args()

face_list_f = list(Path(args.female).rglob('*.npy'))

face_list_m = list(Path(args.male).rglob('*.npy'))

scaler = StandardScaler()

total_testing_crossdb = []

path_f = os.path.join(args.database, 'female')

path_m = os.path.join(args.database, 'male')

test_db_cross_f = list(Path(path_f).rglob('*.npy'))

feat_test_f = np.asarray([np.load(h) for h in test_db_cross_f])

test_db_cross_m = list(Path(path_m).rglob('*.npy'))

feat_test_m = np.asarray([np.load(h) for h in test_db_cross_m])

total_good_f = 0

total_good_m = 0

total_error_f = 0

total_error_m = 0

total_correct = 0

total_error = 0

fpath_csv_th = os.path.join(args.output, "training_{}_testing_{}_knn_shuffle.csv".format(args.train,args.testing))

with open(fpath_csv_th, 'w', newline='') as f:

    fieldnames = ['Train','Test', 'Females_correct', 'Females_error','Males_correct', 'Males_error', 'Total_correct', 'Total_error']

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    path_face_list_f = list(Path(args.female).rglob('*.npy'))

    path_face_list_m = list(Path(args.male).rglob('*.npy'))

    face_list_f = np.asarray([np.load(h) for h in path_face_list_f])
    
    face_list_f = scaler.fit_transform(face_list_f)

    face_list_m = np.asarray([np.load(h) for h in path_face_list_m])

    face_list_m = scaler.fit_transform(face_list_m)

    feat_test_f = scaler.transform(feat_test_f)

    feat_test_m = scaler.transform(feat_test_m)

    labels_f = ['f']*len(path_face_list_f)

    labels_m = ['m']*len(path_face_list_m)

    total_train = np.concatenate((face_list_f,face_list_m),axis=0)

    total_train_label = np.concatenate((labels_f,labels_m),axis=None)

    train_tmp = list(zip(total_train, total_train_label))

    random.shuffle(train_tmp)

    total_train, total_train_label = zip(*train_tmp)

    model = KNeighborsClassifier(n_neighbors=3)

    model_trained = model.fit(total_train,total_train_label)

    prediction_m = model_trained.predict(feat_test_m)

    prediction_f = model_trained.predict(feat_test_f)

    good_males = 0

    for m in prediction_m:

        if m[0] == 'm':

            good_males+= 1

    percent_good_males = (good_males * 100) / len(test_db_cross_m)

    error_males = 100 - percent_good_males
    
    good_females = 0

    for f in prediction_f:

        if f[0] == 'f':

            good_females+= 1

    percent_good_females = (good_females * 100) / len(test_db_cross_f)


    error_females = 100 - percent_good_females

    total_correct = (percent_good_females + percent_good_males) / 2


    writer.writerow({'Train': args.train, 'Test': args.testing,'Females_correct': percent_good_females , 'Males_correct': percent_good_males, 'Total_correct' : total_correct })





























