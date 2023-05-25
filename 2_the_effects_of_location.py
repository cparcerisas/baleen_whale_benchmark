import json
import pathlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dataset
import training
import tensorflow as tf
from sklearn.metrics import confusion_matrix


classes_list = ['20Hz20Plus', 'ABZ', 'DDswp', 'Noise']

output_path = input('Select the folder with the blocked location:')
output_path2 = input('Select the folder with not blocked location (but same strategy!):')

output_path2 = pathlib.Path(output_path2)
f = open(output_path2.joinpath('config.json'))
config = json.load(f)
ds = dataset.SpectrogramDataSet(data_dir=config['DATA_DIR'], image_width=training.IMAGE_WIDTH,
                                image_height=training.IMAGE_HEIGHT,
                                categories=config['CATEGORIES'], join_cat=config["CATEGORIES_TO_JOIN"],
                                locations=config['LOCATIONS'], n_channels=training.N_CHANNELS,
                                corrected=config['USE_CORRECTED_DATASET'],
                                samples_per_class=config['SAMPLES_PER_CLASS'])

output_path = pathlib.Path(output_path)
results = pd.read_csv(output_path.joinpath('total_confusion_matrix.csv'))

results['tpr'] = np.nan
results['detections'] = results[classes_list].sum(axis=1)
for i, row in results.iterrows():
    if row.detections != 0:
        results.loc[i, 'tpr'] = row[row.label] / row.detections

results['fpr'] = np.nan
for fold_n, fold in results.groupby('fold'):
    for i, row in fold.iterrows():
        tp = row[row.label]
        fp = fold[row.label].sum() - tp
        fold_without_label = fold[classes_list].loc[fold.label != row.label]
        total_negative = fold_without_label.sum().sum()

        results.loc[i, 'fpr'] = fp/total_negative

results = results.sort_values(by='fold')
chart = sns.barplot(data=results, x='fold', y='tpr', hue='label')
plt.ylabel('True Positive Rate')
plt.xlabel('Location')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()

chart = sns.barplot(data=results, x='fold', y='fpr', hue='label')
plt.ylabel('False Positive Rate')
plt.xlabel('Location')
plt.ylim([0, 0.8])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()


scores_not_blocked = pd.read_csv(output_path2.joinpath('total_scores.csv'))
scores_blocked = pd.read_csv(output_path.joinpath('total_scores.csv'))

scores_blocked['approach'] = 'Blocked'
scores_not_blocked['approach'] = 'Not blocked'
comparison_results = pd.concat([scores_blocked, scores_not_blocked], ignore_index=True)
sns.boxplot(data=comparison_results, x='approach', y='accuracy')
plt.ylabel('Accuracy')
plt.xlabel('')
plt.show()


test_ds = pd.DataFrame()
for fold in np.arange(config['TEST_SPLIT']):
    data_used_path = output_path2.joinpath('data_used_fold%s_noise%s.csv' % (fold, config['NOISE_RATIO']))
    data_used = pd.read_csv(data_used_path)
    test_img_list = data_used.loc[data_used.set == 'test'].path
    images, labels, paths_list = ds.load_from_file_list(test_img_list)

    model_path = output_path2.joinpath('fold%s' % fold, 'model')
    model = tf.keras.models.load_model(model_path)

    locations = []
    x = ds.reshape_images(images)
    for img_path in paths_list:
        location = img_path.split('_')[1][:-4]
        locations.append(location)

    y_pred = np.argmax(model.predict(x), axis=1)

    test_ds_fold = pd.DataFrame({'path': test_img_list, 'label': labels, 'prediction': y_pred, 'location': locations})

    test_ds = pd.concat([test_ds, test_ds_fold], ignore_index=True)

results_by_location = pd.DataFrame()
# Re-organize by location
for location, test_location in test_ds.groupby('location'):
    y_test = test_location['label'].values
    y_pred = test_location['prediction'].values
    con_mat = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes_list)))

    con_mat_df = pd.DataFrame(con_mat, index=ds.int2class, columns=ds.int2class)
    con_mat_df['fold'] = location
    con_mat_df = con_mat_df.reset_index(names='label')
    results_by_location = pd.concat([results_by_location, con_mat_df], ignore_index=True)


results_by_location['tpr'] = np.nan
results_by_location['detections'] = results_by_location[classes_list].sum(axis=1)
for i, row in results_by_location.iterrows():
    if row.detections != 0:
        results_by_location.loc[i, 'tpr'] = row[row.label] / row.detections

results_by_location['fpr'] = np.nan
for fold_n, fold in results_by_location.groupby('fold'):
    for i, row in fold.iterrows():
        tp = row[row.label]
        fp = fold[row.label].sum() - tp
        fold_without_label = fold[classes_list].loc[fold.label != row.label]
        total_negative = fold_without_label.sum().sum()

        results_by_location.loc[i, 'fpr'] = fp/total_negative

chart = sns.barplot(data=results_by_location, x='fold', y='tpr', hue='label')
plt.ylabel('True Positive Rate')
plt.xlabel('Location')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()

chart = sns.barplot(data=results_by_location, x='fold', y='fpr', hue='label')
plt.ylabel('False Positive Rate')
plt.xlabel('Location')
plt.ylim([0, 0.8])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()