import sys

import numpy as np
import pandas as pd


def tag_label_feature_split(df):
    """
    Returns the tags, labels and features from an MTG_Jamendo working dataset.
    
    A working MTG_Jamendo dataset contains three clusters of fields.  Metadata tags are 
    reflected in columns with names starting with "tag", labels are reflected in columns
    with names starting with "genre", and features are contained in all other columns.

    This function takes an MTG_Jamendo dataset and splits it into three pandas DataFrames, for
    the tags, labels and features present in the dataset, respectively.

    Parameters
    ----------
    df: a pandas DataFrame
        A working MTG_Jamendo dataset, typically created by helpers.reduce.load_and_distill().

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

    return df[tags].copy(), df[labels].copy(), df[features].copy()
