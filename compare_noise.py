import training
import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

if __name__ == '__main__':
    noise_to_check = np.arange (2.5, 100, 2.5).tolist()
    # Read the config file
    config_file = './config.json'
    f = open(config_file)
    config = json.load(f)

    noise_results = pd.DataFrame()
    confusion = []
    for noise in noise_to_check:
        config['NOISE_RATIO'] = noise/100
        scores_df, con = training.run_from_config(config, logpath=pathlib.Path(config['OUTPUT_DIR']).joinpath('noise_%s' % noise))
        scores_df['noise'] = noise
        noise_results = pd.concat([noise_results, scores_df])
        confusion.append(con.diagonal().tolist())
    plt.figure(figsize=(8, 8))
    confusion = np.asanyarray(confusion)
    col = ['tomato','royalblue','mediumseagreen','dimgray']
    for i,c in enumerate(confusion.T):
        plt.scatter(noise_to_check, c,color = col[i],alpha=0.4,label= list(config[ "CATEGORIES_TO_JOIN"].keys())[i])
        xnew=np.arange(min((noise_to_check)),max(noise_to_check),1e-1)
        if len(noise_to_check)>3:
            f=interpolate.UnivariateSpline(noise_to_check,c)
            plt.plot(xnew,f(xnew),'-',color = col[i])
        else:
            g=interpolate.interp1d(noise_to_check,c)
            plt.plot(xnew,g(xnew),'-',color = col[i])
    plt.ylabel('True positive rate')
    plt.xlabel('Noise Percentage')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(pathlib.Path(config['OUTPUT_DIR']).joinpath('noise_categories.png')) 
    plt.show()
    confusion2 = np.mean(confusion[:,:-1], axis=1)
    confusion2 = np.c_[ confusion2, confusion[:,-1] ]

    col = ['orange','dimgray']
    l = ["Calls","Noise"]
    plt.figure(figsize=(8, 8))
    for i,c in enumerate(confusion2.T):
        plt.scatter(noise_to_check, c,color = col[i],label= l[i],alpha=0.4)
        xnew=np.arange(min((noise_to_check)),max(noise_to_check),1e-1)
        if len(noise_to_check)>3:
            f=interpolate.UnivariateSpline(noise_to_check,c)
            plt.plot(xnew,f(xnew),'-',color = col[i])
        else:
            g=interpolate.interp1d(noise_to_check,c)
            plt.plot(xnew,g(xnew),'-',color = col[i])
    plt.ylabel('True positive rate')
    plt.xlabel('Noise Percentage')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(pathlib.Path(config['OUTPUT_DIR']).joinpath('noise_calls.png')) 
    plt.show() 
    noise_results.to_csv(pathlib.Path(config['OUTPUT_DIR']).joinpath('noise_comparison.csv'))
