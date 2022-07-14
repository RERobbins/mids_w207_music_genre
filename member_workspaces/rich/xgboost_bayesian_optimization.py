import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization


sys.path.append("../../")

from helpers.split import make_train_test_split, tag_label_feature_split
from helpers.assess import make_classification_report, make_confusion_matrix

DATASET_FOLDER = "../../datasets/"


def get_data():
    # read a dataset
    df = pd.read_pickle(DATASET_FOLDER + "dataset_00_all.pickle")

    # get labels, a label encoder and features
    _, (y, le), X = tag_label_feature_split(df, label_format="encoded")

    # split the data for training and testing with shuffling and stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1962, shuffle=True, stratify=y
    )
    
    #scaler=StandardScaler()
    
    #pca=KernelPCA(random_state=1962, n_components=200, n_jobs=-1)
    pca=PCA(random_state=1962, n_components=.95)
    
    #pipe = Pipeline([('scaler', scaler),
    #                 ('pca', pca)
    #                ])

    #No scaling -- just PCA
    pipe=pca
    
    # run the preprocessing pipe
    X_train_processed = pipe.fit_transform(X_train)
    X_test_processed = pipe.transform(X_test)
   
    return X_train_processed, y_train


def xgboost_cv(
    features,
    labels,
    learning_rate,
    n_estimators,
    max_depth,
    gamma,
    reg_alpha,
):

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
        seed=1962
    )

    cval = cross_val_score(
        estimator, features, labels, scoring="matthews_corrcoef", cv=5
    )
    return cval.mean()


def optimize_xgboost(features, labels):
    def xgboost_crossval(
        learning_rate, n_estimators, max_depth, gamma, reg_alpha):

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
    optimizer.maximize()

    print("Final result:", optimizer.max)


if __name__ == "__main__":
    features, labels = get_data()

    print("--- Optimizing XGBoost ---")
    optimize_xgboost(features, labels)
