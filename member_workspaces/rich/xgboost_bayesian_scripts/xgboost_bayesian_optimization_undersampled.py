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
from sklean.metrics import accuracy_score, balanced_accuracy_score, f1_score, 

from xgboost import XGBClassifier
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

    # undersample to create balanced dataset
    rus = RandomUnderSampler(random_state=1962)
    X_res, y_res = rus.fit_resample(X, y)
    
    # split into train/validation and test datasets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_res, y_res, test_size=0.2, shuffle=True, stratify=y_res, random_state=1962
    )
    
    # split training/validation into training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, shuffle=True, stratify=y_train_val, random_state=1962
    )
    
    # standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


def xgboost_cv(
    train_features,
    train_labels,
    eval_features,
    eval_labels,
    learning_rate,
    n_estimators,
    max_depth,
    gamma,
    reg_alpha,
    seed=1962,
):

    arguments = locals()
    arguments.pop("train_features")
    arguments.pop("train_labels")
    arguments.pop("eval_features")
    arguments.pop("eval_labels")

    # set the global parameter tags for this exercise
    # for this script, scaling takes place earlier in the process
    # so we just flag that here and do nothing, we aren't using PCA
    # or SMOTE, so we just update the globals
    
    experiment_parameters["scaled"] = "yes"
    experiment_parameters["pca_components"] = 0
    experiment_parameters["resampling"] = "none"
    

    # build out the pipeline depending on the arguments received
    # the final stage is the classifier itself
    # including SMOTE related resampling in the pipeline keeps us from applying SMOTE to
    # validation data, see https://imbalanced-learn.org/stable/common_pitfalls.html#data-leakage
   
    classifier = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma,
        reg_alpha=reg_alpha,
        use_label_encoder=False,
        tree_method="gpu_hist",
        sampling_method="gradient_based",
        objective="multi:softprob",
        eval_metric=["mlogloss", "auc"]
        early_stopping_rounds=10,
        seed=seed,
    )

    eval_set = [(train_features, train_labels), (eval_features, eval_labels)]
    
    classifier.fit(train_features, train_labels, eval_set=eval_set, verbose=False)

    validation_predictions = classifier.predict(eval_features)
    
    score_summary = {"matthews_corrcoef": matthews_corrcoef(eval_labels, validation_predictions),
                     "f1_macro": f1_score(eval_labels, validation_predictions, average='macro'),
                     "balanced_accuracy": balanced_accuracy_score(eval_labels, validation_predictions),
                     "accuracy": accuracy_score(eval_labels, validation_predictions)
                    }

    with open("bayesian_optimization_undersampling_logs.json", "a") as outfile:
        json.dump({**experiment_parameters, **arguments, **score_summary}, outfile)
        outfile.write("\n")

    result = score_summary['matthews_corrcoef']

    return result


def optimize_xgboost(train_features, train_labels, eval_features, eval_labels):
    def xgboost_crossval(learning_rate, n_estimators, max_depth, gamma, reg_alpha):

        return xgboost_cv(
            train_features=train_features,
            train_labels=train_labels,
            eval_feature=eval_features,
            eval_labels=eval_labels,
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            gamma=gamma,
            reg_alpha=reg_alpha,
        )

    optimizer = BayesianOptimization(
        f=xgboost_crossval,
        pbounds={
            "learning_rate": (0.01, 1.0),
            "n_estimators": (100, 1000),
            "max_depth": (3, 10),
            "gamma": (0, 5),
            "reg_alpha": (0, 20),
        },
        random_state=1962,
        verbose=2,
    )

    optimizer.maximize(n_iter=50, init_points=5)
    print("Final result:", optimizer.max)


if __name__ == "__main__":

    dataset = "dataset_00_all.pickle"

    experiment_parameters["model"] = "XGBoostClassifier"
    experiment_parameters["dataset"] = dataset
    experiment_parameters["timestamp"] = int(time.time())

    print("--- Optimizing XGBoost ---")
        print(f"Dataset: {dataset} Arguments")
        train_features, eval_feature, test_features, train_labels, eval_labels, test_labels = get_data(dataset)
        optimize_xgboost(train_features, train_labels, eval_features, eval_labels)
