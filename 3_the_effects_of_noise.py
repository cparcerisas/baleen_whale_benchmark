import json
import pathlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


output_path = input('Select the folder with the noise increment output:')
output_path = pathlib.Path(output_path)
results = pd.read_csv(output_path.joinpath('total_confusion_matrix.csv'))

classes_list = ['20Hz20Plus', 'ABZ', 'DDswp', 'Noise']


results['tpr'] = np.nan
results['detections'] = results[classes_list].sum(axis=1)
for i, row in results.iterrows():
    if row.detections != 0:
        results.loc[i, 'tpr'] = row[row.label] / row.detections

results['fpr'] = np.nan
for fold_n, fold in results.groupby(['fold', 'noise_percentage']):
    for i, row in fold.iterrows():
        tp = row[row.label]
        fp = fold[row.label].sum() - tp
        fold_without_label = fold[classes_list].loc[fold.label != row.label]
        total_negative = fold_without_label.sum().sum()

        results.loc[i, 'fpr'] = fp/total_negative


sns.lmplot(x="noise_percentage", y="tpr", hue="label", data=results)
plt.xlabel('Noise percentage in test')
plt.ylabel('True Positive Rate')
plt.show()


sns.lmplot(x="noise_percentage", y="fpr", hue="label", data=results)
plt.xlabel('Noise percentage in test')
plt.ylabel('False Positive Rate')
plt.show()
