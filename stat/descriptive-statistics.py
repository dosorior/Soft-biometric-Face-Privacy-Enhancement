import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.stats import describe
import os
import math
import csv


plt.rc("axes", axisbelow=True)

parser = argparse.ArgumentParser(description='Computing descriptive statistics',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

parser.add_argument('--input', '-i', 
                    type=str, 
                    help='path where are the databases target')

parser.add_argument('--output', '-o', 
                    type=str, 
                    help='output')

parser.add_argument('--namecol', '-nc', 
                    type=str, 
                    default='maximum',  
                    help='name of the column which will be analysed')

parser.add_argument('--databasef', '-dbf', 
                    type=str, 
                    help='database containing the data corresponding to the female group')

parser.add_argument('--databases', '-dbs', 
                    type=str, 
                    default='_all_against_male_',
                    help='database containing the data corresponding to the male group')

parser.add_argument('--namedb', '-ndb', 
                    type=str, 
                    default='LFW',
                    help='name of the database')

parser.add_argument('--alg', '-algo', 
                    type=str, 
                    default='PRFNet',
                    help='name of the algorithm used for the protection or embedding extractor')                  


parser.add_argument('--savetxt', '-txt', 
                    type=str, 
                    help='path where the file will be saved')

args = parser.parse_args()

#### Settings of the algorithm and databases####

if args.alg == 'PRFNet':

    if args.namedb == 'Adience':

        list_names_comp = []

        list_names_comp.append('CelebA_plus')

        list_names_comp.append('LFW_plus')


    if args.namedb == 'LFW':

        list_names_comp = []

        list_names_comp.append('Adience_plus') 

    if args.namedb == 'CelebA':

        list_names_comp = []

        list_names_comp.append('Adience_plus')

else:

    if args.namedb == 'LFW':

        list_names_comp = []

        list_names_comp.append('Adience_plus')

        list_names_comp.append('FERET_plus')
    
    if args.namedb == 'FERET':

        list_names_comp = []

        list_names_comp.append('Adience_plus')

        list_names_comp.append('LFW_plus')
    
    if args.namedb == 'Adience':

        list_names_comp = []

        list_names_comp.append('LFW_plus')

        list_names_comp.append('FERET_plus')

#### Settings of the algorithm and databases####


fpath_csv_th = os.path.join(args.output, "stat_descriptive_{}.csv".format(args.alg))

with open(fpath_csv_th, 'w', newline='') as f:

    fieldnames = ['db','min_f', 'min_m', 'max_f', 'max_m', 'std_f', 'std_m',  'mean_f', 'mean_m', 'median_f', 'median_m', 'skew_f', 'skew_m', 'kurt_f', 'kurt_m']

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    for db in list_names_comp:

        database_name_f = args.namedb + args.databasef + db + '.csv'

        database_name_m = args.namedb + args.databases + db + '.csv'

        path_data_female = os.path.join(args.input, database_name_f)

        path_data_male = os.path.join(args.input, database_name_m)

        list = [1]

        init_db = database_name_f.split('_')[0] + '_' + database_name_f.split('_')[4]

        for e in list:

            name_col = args.namecol + "{}".format(e)

            id_db = init_db + "_{}".format(e)

            #########Maximum scores######

            data_maximum_female = pd.read_csv(path_data_female, usecols= ['identity','gender', name_col])

            frame_maximum_female = pd.DataFrame(data_maximum_female)

            ##Analysing Males in CelebA##

            data_maximum_male = pd.read_csv(path_data_male, usecols= ['identity','gender', name_col])

            frame_maximum_male = pd.DataFrame(data_maximum_male)

            maximum_celebA_male = frame_maximum_male[name_col]

            maximum_celebA_female = frame_maximum_female[name_col]

            ###Finding the best threshold after subtraction###

            maximum_celebA_male = np.asarray(maximum_celebA_male)

            maximum_celebA_female = np.asarray(maximum_celebA_female)

            scores_subtracted = []

            for value1,value2 in zip(maximum_celebA_female,maximum_celebA_male):

                result_subtracted = np.float32(value1 - value2)

                scores_subtracted.append(np.round(result_subtracted, 6))

            scores_subtracted = np.asarray(scores_subtracted)

            frame_maximum_male['Subtraction'] = scores_subtracted

            data_female = frame_maximum_male[frame_maximum_male.gender == 'f']

            max_subtracted_scores_f = data_female ['Subtraction']

            data_male = frame_maximum_male[frame_maximum_male.gender == 'm']

            max_subtracted_scores_m = data_male ['Subtraction']

            max_subtracted_scores_f = np.asarray(max_subtracted_scores_f)

            max_subtracted_scores_m = np.asarray(max_subtracted_scores_m)

            info_descriptive_f = describe(max_subtracted_scores_f)

            min_score_f = info_descriptive_f.minmax[0]

            max_score_f = info_descriptive_f.minmax[1]

            standard_dev_f = math.sqrt(info_descriptive_f.variance)

            mean_f = info_descriptive_f.mean

            median_f = np.median(max_subtracted_scores_f)

            skewness_f = info_descriptive_f[4]

            kurtosis_f = info_descriptive_f[5]

            info_descriptive_m = describe(max_subtracted_scores_m)

            min_score_m = info_descriptive_m.minmax[0]

            max_score_m = info_descriptive_m.minmax[1]

            standard_dev_m = math.sqrt(info_descriptive_m.variance)

            mean_m = info_descriptive_m.mean

            median_m = np.median(max_subtracted_scores_m)

            skewness_m = info_descriptive_m[4]

            kurtosis_m = info_descriptive_m[5]

            writer.writerow({'db': id_db, 'min_f': np.float32(min_score_f),'min_m': np.float32(min_score_m) , 'max_f': np.float32(max_score_f), 'max_m': np.float32(max_score_m), 'std_f' : np.float32(standard_dev_f), 'std_m': np.float32(standard_dev_m), 'mean_f': np.float32(mean_f), 'mean_m': np.float32(mean_m), 'median_f':np.float32(median_f), 'median_m':np.float32(median_m), 'skew_f': np.float32(skewness_f),'skew_m': np.float32(skewness_m), 'kurt_f': np.float32(kurtosis_f), 'kurt_m': np.float32(kurtosis_m)})






























