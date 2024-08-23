import re
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import argparse
from scipy.stats import describe
import os
from scipy.optimize import brentq
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
import csv



plt.rc("axes", axisbelow=True)

def findIntersection(fun1, fun2, lower, upper):

    return brentq(lambda x : fun1(x) - fun2(x), lower, upper)

# Fit KDE
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return kde_skl, np.exp(log_pdf)

parser = argparse.ArgumentParser(description='Computing false match in biometric system',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

parser.add_argument('--input', '-i', 
                    type=str, 
                    help='input of the file')

parser.add_argument('--output', '-o', 
                    type=str, 
                    help='output')

parser.add_argument('--comparison', '-c', 
                    type=str, 
                    help='name of the file containing the comparison scores')


args = parser.parse_args()

file_read_path = os.path.join(args.input, "{}.csv".format(args.comparison))

file_save_path = os.path.join(args.output, "{}_false_acceptance.csv".format(args.comparison.split('_')[1]+args.comparison.split('_')[3]))

with open(file_save_path, 'w', newline='') as f:

    fieldnames = ['Fix_Th','FMR','Total','Female', 'Male']

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    frame = pd.read_csv(file_read_path)

    data = pd.DataFrame(frame)

    max_scores = data['maximum']

    ####PRFNet###

    # list_th = [0.84,0.76,0.61,0.65] #Th_Adience

    # list_th = [0.70,0.66,0.60,0.62] #Th_CelebA

    # list_th = [0.72,0.67,0.61,0.63] #Th_LFW

    ####PRFNet###

    ###PE-MIU###

    list_th = [0.83,0.70,0.42,0.50] #Th_Adience

    # list_th = [0.62,0.47,0.37,0.40] #Th_ColorFeret

    # list_th = [0.53,0.44,0.37,0.39 ] #Th_LFW

    ###PE-MIU###

    list_fmr = [0.001, 0.01, 0.1, 0.5]

    list_false_match_count = []

    False_match_f = []

    False_match_m = []

    for e in list_th:

        scores_good = data.query('maximum > {}'.format(e))

        total_false_match = (len(scores_good) * 100)/len(max_scores)

        list_false_match_count.append(np.round(np.float32(total_false_match),2))

        male = scores_good[scores_good.gender_attacker == 'm']

        scores_good_m = male.query('maximum > {}'.format(e))

        total_false_match_m = (len(scores_good_m) * 100)/len(scores_good)

        False_match_m.append(np.round(np.float32(total_false_match_m),2))
        
        female = scores_good[scores_good.gender_attacker == 'f']

        scores_good_f = female.query('maximum > {}'.format(e))

        total_false_match_f = (len(scores_good_f) * 100)/len(scores_good)

        False_match_f.append(np.round(np.float32(total_false_match_f),2))
    
    total = list(zip(list_th,list_fmr,list_false_match_count, False_match_f,False_match_m))

    for t in total:

        writer.writerow({'Fix_Th': t[0], 'FMR': t[1], 'Total': t[2],'Female': t[3], 'Male': t[4]})


















       