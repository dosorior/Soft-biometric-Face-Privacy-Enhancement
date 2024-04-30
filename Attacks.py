import numpy as np
import os
import sys, random
import argparse
from pathlib import Path
import pickle
from scipy.spatial import distance
import numpy as np
import csv
from scipy.stats import describe
import math
from sklearn.metrics.pairwise import cosine_similarity

def attacks(list_numbers,list_scores):
    
    list_ave_max = []

    list_log_max = []

    list_linear_max = []

    for n in list_numbers:

        scores_max_list = []

        indices_scores_maximum = (-list_scores).argsort()[:n]

        for pos in indices_scores_maximum:   

            scores_max_list.append(list_scores[pos])  

        scores_max_list = np.asarray(scores_max_list)  

        if n == 1:

            list_ave_max.append(0)

        else:

            value_ave_max_n = ave_maximum_n(scores_max_list)

            list_ave_max.append(value_ave_max_n)
        
        value_log_n = weight_log_n(scores_max_list)

        list_log_max.append(value_log_n)

        value_linear_max_n = weight_linear_n(scores_max_list)

        list_linear_max.append(value_linear_max_n)

    return list_ave_max, list_log_max, list_linear_max   


def ave_maximum_n(scores_max_list):

    ave_max = 0

    ave_max = np.mean(np.float32(scores_max_list))

    return ave_max



def weight_log_n(scores_max_list):

    log_max = 0

    rank = 0

    for e in scores_max_list:

        rank+=1

        log_max += (-(math.log10((1/(rank + 1)) * rank))) * e

    return log_max

def weight_linear_n(scores_max_list):

    linear_max = 0

    rank = 0

    for e in scores_max_list:

        rank+=1

        linear_max += (1 - (1/(rank + 1)) * rank) * e   
        
        #  (((1/(rank + 1)) * rank) * e)


    return linear_max



parser = argparse.ArgumentParser(description='Attack-privacy-template',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-tm', '--testmale', help='path to the male face privacy-enhanced templates', type=str )

parser.add_argument('-tf', '--testfemale', help='path to the female face privacy-enhanced templates', type=str )

parser.add_argument('-e', '--enrol', help= 'path corresponding to the templates to be enrolled or attacked', type=str )

parser.add_argument('-n', '--name', help='soft-biometric to attack, i.e. female or male', type=str)

parser.add_argument('-dbf', '--dbfirst', help= 'name of the database to evaluate at search (e.g. the attacker) for cross-database evaluation' , type=str,
                     default='Adience')      

parser.add_argument('-dbs', '--dbsecond', help='name of the database to evaluate at enrolment (e.g. the attacked) for cross-database evaluation', type=str,
                     default='LFW')               

parser.add_argument('-o', '--output', help='path to the output, file csv with the statistics in terms of similarity scores', type=str)

args = parser.parse_args()


fpath_csv_th = os.path.join(args.output, "{}_all_against_{}_{}_overall_DbBalanced.csv".format(args.dbfirst, args.name, args.dbsecond))

with open(fpath_csv_th, 'w', newline='') as f:

    fieldnames = ['NameAttacker', 'gender','max_1','NameTarget','GenderTarget','max_ave_5', 'max_ave_10', 'max_ave_50', 'max_ave_100', 'max_ave_200', 'max_ave_400', 'max_ave_600','log_1','log_5','log_10','log_50','log_100','log_200','log_400','log_600', 'linear_1', 'linear_5','linear_10','linear_50','linear_100','linear_200','linear_400','linear_600']

    # columns = ['identity', 'gender', 'minimum']

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    
    # writer = csv.writer(f)

    male_feat_lfw = list(Path(args.testmale).glob('*npy'))

    female_feat_lfw = list(Path(args.testfemale).glob('*npy'))

    female_feat_celebA = list(Path(args.enrol).glob('*npy'))

    names_test_list = []

    feat_test_list = []

    list_gender = []

    feat_enrol_list = []

    for feat in male_feat_lfw:

        feat_test_list.append(np.load(str(feat)))

        names_test_list.append(feat.stem)        

        list_gender.append('m')

    for feat in female_feat_lfw:

        feat_test_list.append(np.load(str(feat)))

        names_test_list.append(feat.stem)        

        list_gender.append('f')
    
    names_target = []

    list_gender_target = []
    
    for feat in female_feat_celebA:

        feat_enrol_list.append(np.load(str(feat)))
        
        names_target.append(feat.stem)

        if 'female' in feat.parent.name:
            
            list_gender_target.append('f')

        else:

            list_gender_target.append('m')


    list_numbers = [1,5,10,50,100,200,400,600]

    # list_numbers.append(1)

    # list_numbers.append(5)

    # list_numbers.append(10)

    # list_numbers.append(50)

    # list_numbers.append(100)

    # list_numbers.append(200)

    # list_numbers.append(400)

    # list_numbers.append(600)


       
    for feat_test, gender, name in zip(feat_test_list, list_gender, names_test_list):

        list_scores = []

        list_names = []

        list_g = []

        for feat_enrol, name_t in zip(feat_enrol_list, names_target ):

            list_scores.append(np.float32(1-(distance.cosine(feat_test,feat_enrol))))

            list_names.append(name_t)

            # value_scikyt_reshape = cosine_similarity(feat_test.reshape(1,-1), feat_enrol.reshape(1,-1))[0]

            # value_scipy = float(1 - distance.cosine(feat_test, feat_enrol))

            # value_1 = cosine_similarity((np.asarray(feat_test),np.asarray(feat_enrol)))

            # list_scores.append(np.float32(cosine_similarity(feat_test.reshape(1,-1), feat_enrol.reshape(1,-1))[0]))

        list_scores = np.asarray(list_scores)

        mayor = 0

        info_mayor = " "

        for e,n in zip(list_scores,list_names):

            actual = e

            info_actual = n

            if actual > mayor:
                
                mayor = actual

                info_mayor = info_actual

        list_scores_sort = np.sort(list_scores)[::-1]

        # info_descriptive = describe(list_scores)

        # max_1 = info_descriptive.minmax[1]

        list_ave_max, list_log_max, list_linear_max  = attacks(list_numbers,list_scores)

        writer.writerow({'NameAttacker' : name, 'gender' : gender, 'max_1': np.float32(mayor),'NameTarget' : info_mayor, 'GenderTarget': list_gender_target[0],'max_ave_5': np.float32(list_ave_max[1]), 'max_ave_10': np.float32(list_ave_max[2]), 'max_ave_50': np.float32(list_ave_max[3]), 'max_ave_100': np.float32(list_ave_max[4]), 'max_ave_200': np.float32(list_ave_max[5]), 'max_ave_400' : np.float32(list_ave_max[6]), 'max_ave_600': np.float32(list_ave_max[7]),'log_1': np.float32(list_log_max[0]),'log_5': np.float32(list_log_max[1]),'log_10': np.float32(list_log_max[2]),'log_50': np.float32(list_log_max[3]),'log_100' : np.float32(list_log_max[4]),'log_200': np.float32(list_log_max[5]),'log_400': np.float32(list_log_max[6]),'log_600': np.float32(list_log_max[7]), 'linear_1': np.float32(list_linear_max[0]), 'linear_5': np.float32(list_linear_max[1]),'linear_10': np.float32(list_linear_max[2]),'linear_50': np.float32(list_linear_max[3]),'linear_100': np.float32(list_linear_max[4]),'linear_200': np.float32(list_linear_max[5]),'linear_400': np.float32(list_linear_max[6]),'linear_600': np.float32(list_linear_max[7])})
