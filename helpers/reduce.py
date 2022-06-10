import numpy as np
import pandas as pd

from helpers.stitch import ReadFilesIntoDataframe
from helpers.constants import BASE_GENRES, BASE_FEATURES


def load_and_distill(
    labels=BASE_GENRES, multi_label=False, features=BASE_FEATURES, tags=[], pickle=[]
):

    data = ReadFilesIntoDataframe().read_mtg_jamendo_files()

    if multi_label:
        result = data.drop(data[data[labels].sum(axis=1) == 0].index)
    else:
        result = data.drop(data[data[labels].sum(axis=1) != 1].index)

    if tags == "all":
        tags = result.columns[result.columns.str.startswith("tag")].tolist()

    result = result[tags + features + labels]

    if pickle:
        result.to_pickle(path=pickle, compression="infer")

    return result