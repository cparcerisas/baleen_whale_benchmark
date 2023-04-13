import json
import pathlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


output_path = input('Select the folder with the small dataset output:')
output_path = pathlib.Path(output_path)
f_small = open(output_path.joinpath('config.json'))
config_small = json.load(f_small)
results_small = pd.read_csv(output_path.joinpath('total_confusion_matrix.csv'))
results_small['size'] = 'small'