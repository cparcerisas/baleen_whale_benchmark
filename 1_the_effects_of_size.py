import json
import pathlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

output_path_small = input('Select the folder with the small dataset output:')
output_path_small = pathlib.Path(output_path_small)
f_small = open(output_path_small.joinpath('config.json'))
config_small = json.load(f_small)
results_small = pd.read_csv(output_path_small.joinpath('total_confusion_matrix.csv'))
results_small['size'] = 'small'

output_path_big = input('Select the folder with the big dataset output:')
output_path_big = pathlib.Path(output_path_big)
f_big = open(output_path_small.joinpath('config.json'))
config_big = json.load(f_big)
results_big = pd.read_csv(output_path_big.joinpath('total_confusion_matrix.csv'))
results_big['size'] = 'big'

data = pd.concat([results_small, results_big])
data['true_positive_rate'] = 0
labels = data.label.unique()
for i, row in data.iterrows():
    tp = 0
    fp = 0
    for label in labels:
        if label == row['label']:
            tp = row[label]
        else:
            fp += row[label]
    data['true_positive_rate'] = tp / (tp+fp)

# TODO
# Check that all the parameters are the same except the sample size and the corrected?

sns.violinplot(data=data, x='size', y='true_positive_rate', hue='label')
plt.show()




