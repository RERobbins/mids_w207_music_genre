import numpy as np
import pandas as pd

from helpers.stitch import ReadFilesIntoDataframe


def load_and_distill(multi_label=False):

    data = ReadFilesIntoDataframe().read_mtg_jamendo_files()

    labels = [
        "genre_blues",
        "genre_classical",
        "genre_country",
        "genre_disco",
        "genre_hiphop",
        "genre_jazz",
        "genre_metal",
        "genre_pop",
        "genre_reggae",
        "genre_rock",
    ]

    if multi_label:
        result = data.drop(data[data[labels].sum(axis=1) == 0].index)
    else:
        result = data.drop(data[data[labels].sum(axis=1) != 1].index)
    discard_columns = [
        column
        for column in result.columns
        if ("genre" in column and column not in labels)
    ]
    result.drop(discard_columns, axis=1, inplace=True)
    return result