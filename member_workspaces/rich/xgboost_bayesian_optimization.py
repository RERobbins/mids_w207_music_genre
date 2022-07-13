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



sys.path.append("../../../")

from helpers.split import make_train_test_split, tag_label_feature_split
from helpers.assess import make_classification_report, make_confusion_matrix

DATASET_FOLDER = "../../../datasets/"


def get_data():
    # read a dataset
    df = pd.read_pickle(DATASET_FOLDER + "dataset_01-mean.pickle")
    
    # get labels, a label encoder and features
    _, (y, le), X = tag_label_feature_split(df, label_format="encoded")
    
    # split the data for training and testing with shuffling and stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1962, shuffle=True, stratify=y, 
    )
    
    # XGBoost algorithms do not benefit from scaling, so we won't bother
    
    return X_train, X_test, y_train, y_test
   

def xgboost_cv (features, 
                labels, 
                learning_rate,
                n_estimators,
                max_depth,
                subsample,
                colsample,
                gamma):
    
    estimator = XGBClassifier (
        learning_rate=learning_rate,
        n_estimators=n_estimators,                      
        max_depth=max_depth,
        subsample=subsample,
        colsample=colsample,
        gamma=gamma,
        use_label_encoder=False,
        tree_method='gpu_hist', 
        sampling_method='gradient_based',
        objective="multi:softprob", 
        eval_metric="mlogloss"
    )
    
    cval = cross_val_score(estimator, features, labels, scoring='matthews_corrcoef', cv=5)
    return cval.mean()

def optimize_xgboost(features, labels):

    def xgboost_crossval(learning_rate, n_estimators, max_depth, subsample, colsample, gamma):
        """Wrapper of RandomForest cross validation.

        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return xgboost_cv(
            features=features,
            labels=labels,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            subsample=subsample,
            colsample=colsample,
            gamma=gamma
        )
    
    optimizer = BayesianOptimization(
        f=xgboost_crossval,
        pbounds = {'learning_rate': (0.01, 1.0),
                   'n_estimators': (100, 1000),
                   'max_depth': (3,10),
                   'subsample': (1.0, 1.0),  # Change for big datasets
                   'colsample': (1.0, 1.0),  # Change for datasets with lots of features
                   'gamma': (0, 5)},
        random_state=1962,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)

if __name__ == "__main__":
    features, labels = get_data()

    print("--- Optimizing XGBoost ---"))
    optimize_xgboost(features, labels)