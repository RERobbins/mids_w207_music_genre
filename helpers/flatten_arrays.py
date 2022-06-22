from typing import List
import pandas as pd


def flatten_1d_arrays(df: pd.DataFrame,
                      feature_names: List[str])->pd.DataFrame:
    """Flatten 1d arrays within feature_names
    Parameters:
        df (pd.DataFrame): Original dataframe with fields to be flattened
        feature_names (List[str]): List of column names with 1d arrays
    Returns:
        df (pd.DataFrame): Revised dataframe with flattened 1d array fields
            with the original 1d arrays dropped
    """
    df_flattened_features = []
    for feature_name in feature_names:
        feature_array_shape = df[feature_name][0].shape
        cols_names = [F"{feature_name}_{i}" for i in range(feature_array_shape[0])]
        df_tmp = pd.DataFrame(df[feature_name].tolist(), 
                              index=df.index, 
                              columns=cols_names)
        df_flattened_features.append(df_tmp)
    
    df_flattened_features = pd.concat(df_flattened_features, axis=1)

    df.drop(feature_names, axis=1, inplace=True)
    df = pd.concat([df, df_flattened_features], axis=1)

    return df

def flatten_2d_arrays(df: pd.DataFrame,
                      feature_names: List[str])->pd.DataFrame:
    """Flatten 2d arrays within feature_names
    Parameters:
        df (pd.DataFrame): Original dataframe with fields to be flattened
        feature_names (List[str]): List of column names with 2d arrays
    Returns:
        df (pd.DataFrame): Revised dataframe with flattened 2d array fields
            with the original 2d arrays dropped
    """
    df_flattened_features = []
    for feature_name in feature_names:
        feature_array_shape = df[feature_name][0].shape

        cols_names = [F"{feature_name}_{i}_{j}" \
                      for i in range(feature_array_shape[0]) \
                      for j in range(feature_array_shape[1])]

        df_tmp = pd.DataFrame([i.flatten() for i in df[feature_name].to_list()], 
                              index=df.index, 
                              columns=cols_names)

        df_flattened_features.append(df_tmp)

    df_flattened_features = pd.concat(df_flattened_features, axis=1)
    df.drop(feature_names, axis=1, inplace=True)
    df = pd.concat([df, df_flattened_features], axis=1)

    return df
