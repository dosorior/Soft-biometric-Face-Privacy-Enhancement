import numpy as np
import os
import sys, random
import argparse
from pathlib import Path
import cv2
import pickle
from scipy.spatial import distance
import numpy as np
import csv
from scipy.stats import describe
import math
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='Computing similarity scores from the concept of biometric system',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-m', '--male', type=str,
                    help='loading features corresponding to the group of male from e.g.lfw db')

parser.add_argument('-f', '--female', type=str,
                    help='loading features corresponding to the group of female from e.g.lfw db')


parser.add_argument('-o', '--output', type=str,
                    help='output')

args = parser.parse_args()

fpath_f_f = os.path.join(args.output, "female_female")

fpath_f_m = os.path.join(args.output, "female_male")

fpath_m_m = os.path.join(args.output, "male_male")

fpath_m_f = os.path.join(args.output, "male_female")

dir_male = list(Path(args.male).rglob('*.npy'))

dir_female = list(Path(args.female).rglob('*.npy'))

feat_m = np.asarray([np.load(h) for h in dir_male])

feat_f = np.asarray([np.load(h) for h in dir_female])

label_m = ['male']*len(dir_male)

label_f = ['female']*len(dir_female)

list_scores_f_f,list_scores_f_m,list_scores_m_m,list_scores_m_f  = [],[],[],[]

#f-f#
for b in range(0, len(dir_female)-1):

    temp_f = np.load(str(dir_female[b]))

    for h in range(b+1, len(dir_female)):

        temp =  np.load(str(dir_female[h]))

        value = np.float32(1-(distance.cosine(temp_f,temp)))

        list_scores_f_f.append(np.round(value, 6))

#f-m

for b in range(0, len(dir_female)):

    temp_f = np.load(str(dir_female[b]))

    for h in range(0, len(dir_male)):

        temp = np.load(str(dir_male[h]))

        value = np.float32(1-(distance.cosine(temp_f,temp)))

        list_scores_f_m.append(value)

#m-m
for b in range(0, len(dir_male)-1):

    temp_f = np.load(str(dir_male[b]))

    for h in range(b+1, len(dir_male)):

        temp =  np.load(str(dir_male[h]))

        value = np.float32(1-(distance.cosine(temp_f,temp)))

        list_scores_m_m.append(np.round(value, 6))

#m-f
for b in range(0, len(dir_male)-1):

    temp_f = np.load(str(dir_male[b]))

    for h in range(0, len(dir_female)):

        temp =  np.load(str(dir_female[h]))

        value = np.float32(1-(distance.cosine(temp_f,temp)))

        list_scores_m_f.append(np.round(value, 6))

np.save(fpath_f_f, np.asarray(list_scores_f_f))

np.save(fpath_f_m, np.asarray(list_scores_f_m))

np.save(fpath_m_m, np.asarray(list_scores_m_m))

np.save(fpath_m_m, np.asarray(list_scores_m_f))

