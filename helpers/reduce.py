import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import gc

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import backend as K

from imblearn.under_sampling import RandomUnderSampler

from helpers.stitch import ReadFilesIntoDataframe
from helpers.constants import BASE_GENRES, BASE_FEATURES
from helpers.split import make_train_test_split, tag_label_feature_split


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

def split10(array):
    return np.array_split(array[:, 0:680], 10, axis=1)

def partition(df, shuffle=True):
    df['melspec'] = df['melspec'].apply(split10)
    df = df.explode(['melspec']).reset_index(drop=True)
    if shuffle:
        df = df.sample(frac=1) #shuffle
    
    return df

def transformData(X):
    # extract 2D numpy array from pandas dataframe
    X = np.array(list(X.to_numpy()[:,0]))

    scaler = StandardScaler()
    # flatten 2D array to fit to "overall" mean / variance
    scaler.fit(X.reshape(-1,1))
    # must be reshaped for transformation then restored to original shape
    X = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)

    # add a dimension from (n, 96, 68) to (n, 96, 68, 1)
    X = np.expand_dims(X, axis=-1)

    return X

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def same_prepare_process(df, balanced = False):
    df = partition(df)

    _, (y, le), X = tag_label_feature_split(
        df, label_format="encoded"
    )

    if balanced:
        rus = RandomUnderSampler(random_state=1962)
        X, y = rus.fit_resample(X, y)

    X = transformData(X)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=0.2, random_state=1962, stratify=y
    )

    X_train, X_validation, y_train, y_validation = make_train_test_split(
            X_train, y_train, test_size=0.25, random_state=1962, stratify=y_train
        )

    class_weight={i:c  for i,c in enumerate(compute_class_weight(class_weight='balanced',classes=le.transform(le.classes_),y=y_train))}

    del X
    del y

    gc.collect()

    return X_train, X_test, y_train, y_test, X_validation, y_validation, class_weight

def accuracy_and_loss_curve(hist):
    history = hist.history

    # plot loss for train and validation
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history['loss'], lw=2, color='darkgoldenrod')
    plt.plot(history['val_loss'], lw=2, color='indianred')
    plt.legend(['Train', 'Validation'], fontsize=10)
    ax.set_xlabel('Epochs', size=10)
    ax.set_title('Loss')

    ax = fig.add_subplot(1, 3, 2)
    plt.plot(history['accuracy'], lw=2, color='darkgoldenrod')
    plt.plot(history['val_accuracy'], lw=2, color='indianred')
    plt.legend(['Train', 'Validation'], fontsize=10)
    #plt.ylim(0.7,0.8)
    ax.set_xlabel('Epochs', size=10)
    ax.set_title('accuracy')

    print('After fine-tuning this model, evaluation on the validation data shows an accuracy of:',
        np.round(history['val_accuracy'][-1]*100,2), '%'
    )