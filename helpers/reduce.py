import numpy as np
import pandas as pd
import re

from helpers.stitch import ReadFilesIntoDataframe
from helpers.constants import BASE_GENRES, BASE_FEATURES


def load_and_distill(
    data=None,
    labels=BASE_GENRES,
    multi_label=False,
    features=BASE_FEATURES,
    metadata=[],
    tags=[],
    pickle=[],
):

    """
    Load MTG_Jamendo reference data and reduce it to a working dataset.
    
    Load the sharded MTG_Jamendo reference data and retain elements according 
    to the paramters described below.  Return the result as a pandas DataFrame 
    and if the pickle parameter is specified, save the result as a pickle file, 
    inferring compression from the file name supplied.

    Parameters
    ----------
    data: reference data, if not specified, use the sharded MTG_Jamendo reference data
    
    labels: a list of strings
        The labels (genres) for included tracks, the default is helpers.constants.BASE_GENRES.

    multi_label: a boolean
        If True, a track will be included if it associated with at least one label from the 
        label paramters, otherwise only tracks associated with a single label but not more 
        than one will be included.  The default is False.

    features: 'all', 'all_scalar' or a list of strings
        The features to be included for each track, the default is helpers.constants.BASE_FEATURES.
        If the parameter is 'all', then all features will be included.  The 'all_scalar' value is
        used to specify all features other than non_scalar features.
        
    metadata: 'all' or a list of strings
        The metadata to be included for each track.  If the paramter is the empty list (the default), 
        metadata will not be included.  If the paramter is 'all', then all metadata will be included.

    tags: 'all' or a list of strings
        The tags to be included for each track.  If the paramter is the empty list (the default), 
        tags will not be included.  If the paramter is 'all', then all tags will be included.

    pickle: a string
        If a string is supplied, it will be used as the name of the pickle file for the result.  
        Compression is inferred from the name.  If the paramter is not specified, the pickle 
        file will not be created.

    Returns
    -------
    pandas DataFrame
        The DataFrame specified by the parameters.    
    """

    if data is None:
        data = ReadFilesIntoDataframe().read_mtg_jamendo_files()

    if multi_label:
        result = data.drop(data[data[labels].sum(axis=1) == 0].index)
    else:
        result = data.drop(data[data[labels].sum(axis=1) != 1].index)

    if metadata == "all":
        metadata = [
            column for column in result.columns if column.startswith("metadata")
        ]

    if tags == "all":
        tags = [column for column in result.columns if column.startswith("tag")]

    if features == "all":
        features = [
            column
            for column in result.columns
            if not re.match("tag|genre|metadata", column)
        ]

    if features == "all_scalar":
        all_features = [
            column
            for column in result.columns
            if not re.match("tag|genre|metadata", column)
        ]
        non_scalar = result.select_dtypes(exclude="number")
        all_scalar = set(all_features) - set(non_scalar)
        features = [column for column in result.columns if column in all_scalar]

    result = result[metadata + tags + features + labels]

    if pickle:
        result.to_pickle(path=pickle, compression="infer")

    return result