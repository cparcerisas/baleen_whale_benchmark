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

    noise_results = pd.DataFrame(columns=['noise', 'loss', 'accuracy_test'])
    for noise in noise_to_check:
        config['NOISE_RATIO'] = noise/100
        scores = training.run_from_config(config, logpath=pathlib.Path(config['OUTPUT_DIR']).joinpath('noise_%s' % noise))
        noise_results.loc[len(noise_results)] = [noise, scores[0], scores[1]]

    noise_results.to_csv(config['OUTPUT_DIR'])
