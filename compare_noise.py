import training
import json
import pathlib
import pandas as pd


if __name__ == '__main__':
    noise_to_check = [10, 20, 30, 40, 50, 60, 90, 95, 98]
    # Read the config file
    config_file = './config.json'
    f = open(config_file)
    config = json.load(f)

    noise_results = pd.DataFrame()
    for noise in noise_to_check:
        config['NOISE_RATIO'] = noise/100
        scores_df, con_matrix = training.run_from_config(config,
                                                         logpath=pathlib.Path(config['OUTPUT_DIR']
                                                                              ).joinpath('noise_%s' % noise))
        scores_df['noise'] = noise
        noise_results = pd.concat([noise_results, scores_df])

    noise_results.to_csv(pathlib.Path(config['OUTPUT_DIR']).joinpath('noise_comparison.csv'))
