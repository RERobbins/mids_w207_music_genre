import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

from sklearn import SVC
from bayes_opt import BayesianOptimization

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

sys.path.append("../../")

from helpers.split import tag_label_feature_split

DATASET_FOLDER = "../../datasets/"

experiment_parameters = {}


def get_data(dataset):

    # read a dataset
    df = pd.read_pickle(DATASET_FOLDER + dataset)

    # get labels, a label encoder and features
    _, (y, le), X = tag_label_feature_split(df, label_format="encoded")

    
    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=1962
    )
      
    # standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
       
    return X_train_scaled, X_test_scaled, y_train, y_test


def svm_cv(train_features, train_labels, C, gamma, kernel='rbf'):

    arguments = locals()
    arguments.pop("train_features")
    arguments.pop("train_labels")

    # set the global parameter tags for this exercise
    # for this script, scaling takes place earlier in the process
    # so we just flag that here and do nothing, we aren't using PCA
    # or SMOTE, so we just update the globals
    
    experiment_parameters["scaled"] = "yes"
    experiment_parameters["pca_components"] = 0
 
    classifier = SVC(
        C=C,
        kernel=kernel,
        gamme=gamma,
        class_weight='balanced',
    )
    
    scoring = ['matthews_corrcoef', 'f1_macro', 'balanced_accuracy', 'accuracy']

    scores = cross_validate (classifier, 
                             train_features, 
                             y=y_train_labels, 
                             scoring=scoring, 
                             return_train_score=True, 
                             return_estimator=True,
                             n_jobs=-1)
    
    score_summary = {"matthews_corrcoef": scores['test_matthews_corrcoef'].mean(),
                     "f1_macro": scores['test_f1_macro'].mean(),
                     "balanced_accuracy": scores['test_balanced_accuracy'].mean(),
                     "accuracy": scores['accuracy'].mean()
                    }

    with open("bayesian_optimization_svm_logs.json", "a") as outfile:
        json.dump({**experiment_parameters, **arguments, **score_summary}, outfile)
        outfile.write("\n")

    result = score_summary['matthews_corrcoef']

    return result


def optimize_svm(train_features, train_labels):
    def xgboost_crossval(C, gamma):
        return xgboost_cv(train_features, train_labels, C, gamma)
            
    optimizer = BayesianOptimization(
        f=xgboost_crossval,
        pbounds={
            "C": (.1, 1000),
            "gamma": (0.0001, 1),
        },
        random_state=1962,
        verbose=2,
    )

    optimizer.maximize(n_iter=100, init_points=5)
    print("Final result:", optimizer.max)


if __name__ == "__main__":

    dataset = "dataset_00_all.pickle"

    experiment_parameters["model"] = "SVM"
    experiment_parameters["dataset"] = dataset
    experiment_parameters["timestamp"] = int(time.time())

    print("--- Optimizing XGBoost ---")
    print(f"Dataset: {dataset} Arguments")
    train_features, test_features, train_labels, test_labels = get_data(dataset)
    optimize_xgboost(train_features, train_labels)
