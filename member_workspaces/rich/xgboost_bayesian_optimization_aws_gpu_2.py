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

from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

sys.path.append("../../")

from helpers.split import tag_label_feature_split

DATASET_FOLDER = "../../datasets/"

experiment_parameters = {}


def get_data(dataset):

    # read a dataset
    df = pd.read_pickle(DATASET_FOLDER + dataset)

    # get labels, a label encoder and features
    _, (y, le), X = tag_label_feature_split(df, label_format="encoded")

    # split the data for training and testing with shuffling and stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1962, shuffle=True, stratify=y
    )

    return X_train, y_train


def xgboost_cv(
    features,
    labels,
    scaling,
    pca_components,
    resampling,
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
    arguments.pop("scaling")
    arguments.pop("pca_components")
    arguments.pop("resampling")

    # set the global parameter tags for this exercise
    experiment_parameters["scaled"] = "yes" if scaling else "no"
    experiment_parameters["pca_components"] = (
        0 if pca_components is None else pca_components
    )
    experiment_parameters["resampling"] = "none" if resampling is None else resampling

    # build out the pipeline depending on the arguments received
    # the final stage is the classifier itself
    # including SMOTE related resampling in the pipeline keeps us from applying SMOTE to
    # validation data, see https://imbalanced-learn.org/stable/common_pitfalls.html#data-leakage
   
    steps = []

    if scaling:
        steps.append(("scaling", StandardScaler()))

    if pca_components is not None:
        steps.append(("pca", PCA(random_state=1962, n_components=pca_components)))

    if resampling == "SMOTE":
        steps.append(("resampling", SMOTE(random_state=1962, n_jobs=-1)))

    if resampling == "SMOTEENN":
        enn = EditedNearestNeighbours(kind_sel="mode", n_neighbors=3, n_jobs=-1)
        steps.append(("resampling", SMOTEENN(random_state=1962, enn=enn)))

    if resampling == "SMOTETomek":
        steps.apend(("resampling", SMOTETomek(random_state=1962, n_jobs=-1)))

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
        eval_metric="mlogloss",
        seed=seed,
    )

    steps.append(("classifier", classifier))

    model = Pipeline(steps)

    scoring = ["matthews_corrcoef", "f1_macro", "accuracy"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            model, features, labels, scoring=scoring, cv=10, error_score=0, n_jobs=-1
        )

    score_summary = {key.strip("test_"): value.mean() for key, value in scores.items()}

    with open("bayesian_optimization_aws_gpu_2.logs.json", "a") as outfile:
        json.dump({**experiment_parameters, **arguments, **score_summary}, outfile)
        outfile.write("\n")

    result = scores["test_matthews_corrcoef"].mean()

    return result


def optimize_xgboost(features, labels, scaling, pca_components, resampling):
    def xgboost_crossval(learning_rate, n_estimators, max_depth, gamma, reg_alpha):

        return xgboost_cv(
            features=features,
            labels=labels,
            scaling=scaling,
            pca_components=pca_components,
            resampling=resampling,
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

    arg_scaling = [True]
    arg_pca = [0.95]
    arg_resampling = [None, "SMOTE", "SMOTEENN", "SMOTETOMEK"]
    arg_dicts = [
        {"scaling": scaling, "pca_components": pca_components, "resampling": resampling}
        for scaling in arg_scaling
        for pca_components in arg_pca
        for resampling in arg_resampling
    ]

    for arg_dict in arg_dicts:
        print("--- Optimizing XGBoost ---")
        print(f"Dataset: {dataset} Arguments: {arg_dict}")
        features, labels = get_data(dataset)
        optimize_xgboost(features, labels, **arg_dict)
