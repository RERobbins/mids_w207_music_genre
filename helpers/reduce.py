import numpy as np
import pandas as pd

from helpers.stitch import ReadFilesIntoDataframe
from helpers.constants import BASE_GENRES, BASE_FEATURES


def load_and_distill(
    labels=BASE_GENRES, multi_label=False, features=BASE_FEATURES, tags=[], pickle=[]
):

    """
    Load MTG_Jamendo reference data and reduce it to a working dataset.
    
    Load the sharded MTG_Jamendo reference data and retain elements according 
    to the paramters described below.  Return the result as a pandas DataFrame 
    and if the pickle parameter is specified, save the result as a pickle file, 
    inferring compression from the file name supplied.

    Parameters
    ----------
    labels: a list of strings
        The labels (genres) for included tracks, the default is helpers.constants.BASE_GENRES.

    multi_label: a boolean
        If True, a track will be included if it associated with at least one label from the 
        label paramters, otherwise only tracks associated with a single label but not more 
        than one will be included.  The default is False.

    features: a list of strings
        The features to be included for each track, the default is helpers.constants.BASE_FEATURES.

    tags: 'all' or a list of strings
        The tags to be included for each track.  If the paramter is the empty list (the default), 
        tags will not be included.  If the paramter is 'all', all tags will be included.

    pickle: a string
        If a string is supplied, it will be used as the name of the pickle file for the result.  
        Compression is inferred from the name.  If the paramter is not specified, the pickle 
        file will not be created.

    Returns
    -------
    pandas DataFrame
        The DataFrame specified by the parameters.    
    """

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
