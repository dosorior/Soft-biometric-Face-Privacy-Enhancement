from typing import ValuesView
import numpy as np
import sys
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.arrays.sparse import dtype
import seaborn as sns
import argparse
import os

fig, ax = plt.subplots(figsize=(5, 5))

plt.rc("axes", axisbelow=True)

parser = argparse.ArgumentParser(description='Data Visualization of the Effect of Homogeinity using Boxplots',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

parser.add_argument('--input', '-i', 
                    type=str, 
                    help='path where are the similarity scores to be loaded')

parser.add_argument('--output', '-o', 
                    type=str, 
                    help='output')

args = parser.parse_args()

dir_scores_f_f = os.path.join(args.input, 'female_female.npy')

scores_f_f = np.load(dir_scores_f_f)

scores_f_f = np.sort(scores_f_f)[::-1]

dir_scores_f_m = os.path.join(args.input, 'female_male.npy')

scores_f_m = np.load(dir_scores_f_m)

scores_f_m = np.sort(scores_f_m)[::-1]

dir_scores_m_m = os.path.join(args.input, 'male_male.npy')

scores_m_m = np.load(dir_scores_m_m) 

scores_m_m = np.sort(scores_m_m)[::-1]

dir_scores_m_f = os.path.join(args.input, 'male_female.npy')

scores_m_f = np.load(dir_scores_m_f)

scores_m_f = np.sort(scores_m_f)[::-1]

dataframe = pd.DataFrame(columns= ['Same Attribute', 'Different Attribute'])

similar_attribute = np.concatenate((scores_f_f,scores_m_m),axis=None)

dataframe['Same Attribute'] = similar_attribute

different_attribute = np.concatenate((scores_f_m,scores_m_f),axis=None)

different_attribute = different_attribute[0:1652510]

path_different = os.path.join(args.input, 'different_attribute.npy')

np.save(path_different, np.asarray(different_attribute))

dataframe['Different Attribute'] = different_attribute

labels = ['Same Attribute','Different Attribute']

flierprops = {'marker':'o',
                  'markerfacecolor':'black',
                  'markersize':1,
                  'markeredgecolor':'black',
                #   'alpha':0.5,
                  'linewidth':1.5}
bx = ax.boxplot(dataframe, labels=labels ,flierprops=flierprops)

plt.ylabel("Score")

plt.grid(True)

path_save = os.path.join(args.output, 'LFW.pdf')

plt.savefig(path_save, bbox_inches="tight")

# plt.show()



