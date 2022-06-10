import sys

import numpy as np
import pandas as pd

from helpers.reduce import load_and_distill


def tag_label_feature_split(df):
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