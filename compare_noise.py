import training
import json
import pathlib
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import os
if __name__ == '__main__':

    def fit_polynomials(X, y, from_=1, to_= 10, step=1):
        # Store scores and predictions
        scors = []
        # Loop between the specified values
        X = X[:, np.newaxis]
        y = y[:, np.newaxis]
        xnew=np.arange(min(X),max(X),1e-1)
        xnew = xnew[:, np.newaxis]
        for i, n in enumerate(range(from_, to_+1, step)):
            # Steps
            steps = [
            ('Polynomial', PolynomialFeatures(degree=n)),
            ('model', LinearRegression())  ] 
            # Pipeline fit
            fit_poly = Pipeline(steps).fit(X,y)
            # Predict
            poly_pred = fit_poly.predict(xnew)
            if i == 0:
                preds = pd.DataFrame(poly_pred, columns=[f'{n}'])
            else:
                preds[f'{n}'] = poly_pred
            # Evaluate
            model_score = fit_poly.score(X,y)
            scors.append((n, model_score))
        s_list = [x[1] for x in scors]
        smax = max(s_list)
        max_index = s_list.index(smax)
        df = preds[str(scors[max_index][0])].copy()
        return df,xnew


    noise_to_check = np.arange (2.5, 82.5, 2.5).tolist()
    # Read the config file
    config_file = './config.json'
    f = open(config_file)
    config = json.load(f)

    noise_results = pd.DataFrame()
    confusion = []
    path = pathlib.Path(config['OUTPUT_DIR']).joinpath(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
    os.mkdir(str(path))
    for noise in noise_to_check:
        config['NOISE_RATIO'] = noise/100
        scores_df, con = training.run_from_config(config, logpath=pathlib.Path(path).joinpath('noise_%s' % noise))
        scores_df['noise'] = noise
        noise_results = pd.concat([noise_results, scores_df])
        confusion.append(con.diagonal().tolist())
    plt.figure(figsize=(8, 8))
    confusion = np.asanyarray(confusion)
    col = ['tomato','royalblue','mediumseagreen','dimgray']
    n = np.asanyarray(noise_to_check)
    for i,c in enumerate(confusion.T):
        plt.scatter(noise_to_check, c,color = col[i],alpha=0.4,label= list(config[ "CATEGORIES_TO_JOIN"].keys())[i])
        preds, xnew = fit_polynomials(n, c, from_=1, to_=20, step=1)
        plt.plot(xnew, preds.values,'-',color = col[i])
    plt.ylabel('True positive rate')
    plt.xlabel('Noise Percentage')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(pathlib.Path(path).joinpath('noise_categories.png')) 
    plt.show()
    confusion2 = np.mean(confusion[:,:-1], axis=1)
    confusion2 = np.c_[ confusion2, confusion[:,-1] ]

    col = ['orange','dimgray']
    l = ["Calls","Noise"]
    n = np.asanyarray(noise_to_check)
    plt.figure(figsize=(8, 8))
    for i,c in enumerate(confusion2.T):
        plt.scatter(noise_to_check, c,color = col[i],label= l[i],alpha=0.4)
        preds, xnew = fit_polynomials(n, c, from_=1, to_=5, step=1)
        plt.plot(xnew, preds.values,'-',color = col[i])
    plt.ylabel('True positive rate')
    plt.xlabel('Noise Percentage')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(pathlib.Path(path).joinpath('noise_calls.png')) 
    plt.show() 
    noise_results.to_csv(pathlib.Path(path).joinpath('noise_comparison.csv'))
