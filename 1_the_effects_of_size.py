import json
import pathlib

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import pycm

output_path_small = input('Select the folder with the small uncorrected dataset output:')
output_path_small = pathlib.Path(output_path_small)
results_small = pd.read_csv(output_path_small.joinpath('total_confusion_matrix.csv'))
scores_small = pd.read_csv(output_path_small.joinpath('total_scores.csv'))
scores_small['size'] = 'Small uncorrected'
results_small['size'] = 'Small uncorrected'


output_path_small_corr = input('Select the folder with the small corrected dataset output:')
output_path_small_corr = pathlib.Path(output_path_small_corr)
results_small_corr = pd.read_csv(output_path_small_corr.joinpath('total_confusion_matrix.csv'))
scores_small_corr = pd.read_csv(output_path_small_corr.joinpath('total_scores.csv'))
scores_small_corr['size'] = 'Small corrected'
results_small_corr['size'] = 'Small corrected'


output_path_big = input('Select the folder with the big dataset output:')
output_path_big = pathlib.Path(output_path_big)
results_big = pd.read_csv(output_path_big.joinpath('total_confusion_matrix.csv'))
scores_big = pd.read_csv(output_path_big.joinpath('total_scores.csv'))
scores_big['size'] = 'Large'
results_big['size'] = 'Large'


print('First, plot the average performance')
total_scores = pd.concat([scores_small, scores_big, scores_small_corr], ignore_index=True)
sns.boxplot(total_scores, x='size', y='accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Dataset used')
plt.show()

classes_list = ['20Hz20Plus', 'ABZ', 'DDswp', 'Noise']

total_cross_matrix = pd.concat([results_small, results_big, results_small_corr], ignore_index=True)

total_cross_matrix['tpr'] = np.nan
total_cross_matrix['detections'] = total_cross_matrix[classes_list].sum(axis=1)
for i, row in total_cross_matrix.iterrows():
    total_cross_matrix.loc[i, 'tpr'] = row[row.label] / row.detections

total_cross_matrix['fpr'] = np.nan
for fold_n, fold in total_cross_matrix.groupby(['size', 'fold']):
    for i, row in fold.iterrows():
        tp = row[row.label]
        fp = fold[row.label].sum() - tp
        fold_without_label = fold[classes_list].loc[fold.label != row.label]
        total_negative = fold_without_label.sum().sum()

        total_cross_matrix.loc[i, 'fpr'] = fp/total_negative


sns.violinplot(data=total_cross_matrix, x='size', y='tpr', hue='label')
plt.ylabel('True Positive Rate')
plt.xlabel('Dataset size')
plt.show()

sns.violinplot(data=total_cross_matrix, x='size', y='fpr', hue='label')
plt.ylabel('False Positive Rate')
plt.xlabel('Dataset size')
plt.show()




