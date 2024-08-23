from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from yellowbrick.text import TSNEVisualizer
from yellowbrick.datasets import load_hobbies
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser(description='Data Visualization of the female and male distributions using T-SNE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

parser.add_argument('--input', '-i', 
                    type=str, 
                    help='path where are the face embeddings to be loaded')

parser.add_argument('--output', '-o', 
                    type=str, 
                    help='output')

args = parser.parse_args()

fig, ax = plt.subplots()

# Load the data and create document vectors
corpus = load_hobbies()

tfidf = TfidfVectorizer()

dir_male = list(Path(args.input).rglob('*.npy'))

dir_female = list(Path(args.input).rglob('*.npy'))

feat_m = np.asarray([np.load(h) for h in dir_male])

feat_f = np.asarray([np.load(h) for h in dir_female])

label_m = ['male']*len(dir_male)

label_f = ['female']*len(dir_female)

total_feat = np.concatenate((feat_m, feat_f),axis=0)

total_label = np.concatenate((label_m,label_f),axis=None)

for i in range(1):

    X = total_feat

    Y = total_label

    labels = ['Male', 'Female']

    plt.rc('legend', fontsize=17.5) 

    plt.rc('axes', labelsize=15) 

    plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.rc('font', size=15)          # controls default text sizes
    plt.xlim(-80,80)
    plt.ylim(-80,80)
    ax.tick_params(labelsize=18)
    
    tsne = TSNEVisualizer(ax = ax,decompose='pca', colors=['purple', 'blue'],labels=labels)
    tsne.fit(X, Y)
    # tsne.show()
    h, l = ax.get_legend_handles_labels()
    ax.clear()
    ax.legend(h,l,loc='upper right')
    plt.grid(True)
 
path_save = os.path.join(args.output)
plt.savefig(path_save)







