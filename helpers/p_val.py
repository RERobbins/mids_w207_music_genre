import pandas as pd
import numpy as np
import scipy.stats as stats

from helpers.stitch import ReadFilesIntoDataframe
from helpers.constants import BASE_GENRES
from helpers.split import tag_label_feature_split

def p_val_plots(genre, features, labels):
    genre_features = features[labels[genre] == 1]
    non_genre_features = features[labels[genre] == 0]

    p_vals = []

    for i in genre_features.columns:
        # more than 10 samples for the feature to assume normality for t-test
        # if len(genre_features[genre_features[i] != 0].index) >= 10:
        if len(genre_features.index) >= 10:
            p_vals.append(stats.ttest_ind(genre_features[i], non_genre_features[i], equal_var = False).pvalue)

    # keeping column where p value < 0.05
    features_p_val = pd.DataFrame(features.columns, columns=[genre])
    p_vals = pd.DataFrame(p_vals, columns=['p_val'])
    features_p_val = pd.concat([features_p_val, p_vals], axis=1)
    features_p_val = features_p_val[features_p_val['p_val'] <= 0.05]
    features_no_relation = features_p_val[features_p_val['p_val'] > 0.05]
    features_p_val = features_p_val.sort_values(by='p_val',ignore_index=True)

    return features_p_val, features_no_relation

def get_p_val_ranking():

    read_file = ReadFilesIntoDataframe()
    df = read_file.read_mtg_jamendo_files()

    df = df[df[BASE_GENRES].sum(axis=1) == 1]

    _, labels, features = tag_label_feature_split(df)

    features = features.select_dtypes('float')

    features_ranked = None
    features_no_relation = None

    for i in BASE_GENRES:
        features_p_val, features_no_rel = p_val_plots(i, features, labels)

        if features_ranked is not None:
            features_ranked = pd.concat([features_ranked, features_p_val[i]], axis = 1)
            features_no_relation = pd.concat([features_no_relation, features_no_rel[i]], axis = 1)

        else:
            features_ranked = pd.DataFrame(features_p_val[i].tolist(), columns=[i])
            features_no_relation = pd.DataFrame(features_no_rel[i].tolist(), columns=[i])

    total_ranked = pd.DataFrame({
        'columns': features.columns, 
        'rankings': np.zeros(shape = len(features.columns))
    })

    for i in range(len(features.columns)):
        for g in BASE_GENRES:
            indexes = features_ranked[g][features_ranked[g] == features.columns[i]].index.to_list()
            if len(indexes) > 0:
                index = int(indexes[0])
            else:
                index = 2282

            total_ranked.iloc[[i], [1]] = int(total_ranked.iloc[i]['rankings']) + index

    print(total_ranked.sort_values(by='rankings',ignore_index=True).head(10))