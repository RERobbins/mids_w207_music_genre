import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate

from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

sys.path.append("../../")

from helpers.split import tag_label_feature_split
from helpers.assess import make_classification_report, make_confusion_matrix

DATASET_FOLDER = "../../datasets/"

experiment_parameters = {}

def get_data(dataset, scaling=False, pca_components=None, resampling=None):
    
    # set the global parameter tags for this exercise
    experiment_parameters['dataset'] = dataset
    experiment_parameters['scaled'] = 'yes' if scaling else 'no'
    experiment_parameters['pca_components'] = 0 if pca_components is None else pca_components
    experiment_parameters['resampling'] = "none" if resampling is None else resampling
    
    # read a dataset
    df = pd.read_pickle(DATASET_FOLDER + dataset)

    # get labels, a label encoder and features
    _, (y, le), X = tag_label_feature_split(df, label_format="encoded")

    # split the data for training and testing with shuffling and stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1962, shuffle=True, stratify=y
    )

    # use scaling or pca as required
    pipe=None
    
    if scaling and pca_components is not None:
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('pca', PCA(random_state=1962, n_components=pca_components))])
        
    if scaling and pca_components is None:
        pipe = Pipeline([('scaler', StandardScaler())])
        
    if not scaling and pca_components is not None:
        pipe = Pipeline([('pca', PCA(random_state=1962, n_components=pca_components))])
          
    if pipe is not None:
        X_train = pipe.fit_transform(X_train)    

    # resample as required
    smt = None
    
    if resampling == 'SMOTE':
        smt = SMOTE (random_state=1962, n_jobs=-1)
        
    if resampling == 'SMOTEENN':
        enn = EditedNearestNeighbours(kind_sel='mode', n_neighbors=3, n_jobs=-1)
        smt = SMOTEENN (random_state=1962, enn=enn)

    if resampling == "SMOTETomek":
        smt = SMOTETomek(random_state=1962, n_jobs=-1)
        
    if smt is not None:
        X_train, y_train = smt.fit_resample(X_train, y_train)
        
    return X_train, y_train


def xgboost_cv(
    features,
    labels,
    learning_rate,
    n_estimators,
    max_depth,
    gamma,
    reg_alpha,
    seed=1962,
):

    arguments = locals()
    arguments.pop("features")
    arguments.pop("labels")

    estimator = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma,
        reg_alpha=reg_alpha,
        use_label_encoder=False,
        tree_method="gpu_hist",
        sampling_method="gradient_based",
        objective="multi:softprob",
        eval_metric="mlogloss",
        seed=seed,
    )

    scoring = ["matthews_corrcoef", "f1_macro", "accuracy"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            estimator, features, labels, scoring=scoring, cv=10, error_score=0, n_jobs=-1
        )

    score_summary = {key.strip("test_"): value.mean() for key, value in scores.items()}

    with open("bayesian_optimization_logs.json", "a") as outfile:
        json.dump({**experiment_parameters, **arguments, **score_summary}, outfile)
        outfile.write("\n")

    result = scores["test_matthews_corrcoef"].mean()
    
    return result

def optimize_xgboost(features, labels):
    def xgboost_crossval(learning_rate, n_estimators, max_depth, gamma, reg_alpha):

        return xgboost_cv(
            features=features,
            labels=labels,
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
    
    dataset='dataset_01_mean.pickle'
       
    experiment_parameters['model'] = "XGBoostClassifier"
    experiment_parameters['timestamp'] = int(time.time())

    arg_scaling = [False, True]
    arg_pca = [None, .95]
    arg_resampling = [None, 'SMOTE', 'SMOTEENN', 'SMOTETOMEK']
    arg_dicts = [{'scaling': scaling, 'pca_components': pca_components, 'resampling': resampling}
                 for scaling in arg_scaling
                 for pca_components in arg_pca
                 for resampling in arg_resampling]

    for arg_dict in arg_dicts:
        print("--- Optimizing XGBoost ---")
        print (f"Dataset: {dataset} Arguments: {arg_dict}")
        features, labels = get_data(dataset, **arg_dict)
        optimize_xgboost(features, labels)
