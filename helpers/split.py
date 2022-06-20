import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def tag_label_feature_split(df,label_format='one_hot'):
    """
    Returns the tags, labels and features from an MTG_Jamendo working dataset.
    
    A working MTG_Jamendo dataset contains three clusters of fields.  Metadata tags are 
    reflected in columns with names starting with "tag", labels are reflected in columns
    with names starting with "genre", and features are contained in all other columns.

    This function takes an MTG_Jamendo dataset and splits it into three pandas DataFrames, for
    the tags, labels and features present in the dataset, respectively.

    Parameters
    ----------
    df: pandas DataFrame
        A working MTG_Jamendo dataset, typically created by helpers.reduce.load_and_distill().

    label_format:
        optionally specify "label_strings" to apply the label_strings fn to the one-hot labels,
        returning a dense vector of lable strings instead
      OR
        optionally specify "encoded" to encode the one-hot labels to a dense vector of label
        integers, and return as a tuple along with a label encoder object

    Returns
    -------
    three pandas DataFrames
        tag DataFrame, label DataFrame, features DataFrame    
    """

    tags = []
    labels = []
    features = []

    for name in df.columns:
        if name.startswith("genre"):
            labels.append(name)
        elif name.startswith("tag"):
            tags.append(name)
        else:
            features.append(name)

    labels = df[labels].copy();
    if(label_format=='label_strings' or label_format=='encoded'):
        labels = label_strings(labels)
        if label_format=='encoded':
            le = LabelEncoder()
            labels = le.fit_transform(pd.Series.ravel(labels))
            labels = (labels,le)

    return df[tags].copy(), labels, df[features].copy()


def label_strings(one_hot_encoded_labels):
    """
    Returns a one column dataframe consisting of the label strings corresponding to
    the one hot enoded labels passed to the function.
    
    Parameters
    ----------
    one_hot_encoded_labels: pandas Dataframe
        A pandas DataFrame of one hot encoded labels such as the labels returned by
        tag_label_feature_split().
        
    Returns
    -------
    pandas DataFrame
        A one column pandas DataFrame with the labels strings corresponding to
        the one hot encoded labels passed to the function.  The returned object
        is suitable for use with sklearn.preprocessing.LabelBinarizer.
    """
    
    return one_hot_encoded_labels.idxmax(axis='columns').to_frame(name='label')


# extending the sklearn make_train_test_split to optionally perform X scaling automatically
def make_train_test_split(
  X,
  y,
  test_size=None,
  train_size=None,
  random_state=None,
  shuffle=True,
  stratify=None,
  x_scaler=None # optionally pass sklearn scaler to fit to train data, then apply to train and test data
):
  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    train_size=train_size,
    random_state=random_state,
    shuffle=shuffle,
    stratify=stratify,
  )

  # apply scaler to X data if provided
  x_scaler = resolve_scaler(x_scaler)
  if(x_scaler):
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

# reusable helper function to get an instance of a scaler from various polymorphisms
def resolve_scaler(s):
  if s == 'standard': # shortcut to initialize a stock StandardScaler
    s = StandardScaler()
  if isinstance(s,type): # if constructor is passed, create instance
    s = s()
  return s # (if none, or already initialized scaler, return "self"