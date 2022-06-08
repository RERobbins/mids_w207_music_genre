import os
import re
import pandas as pd

from pathlib import Path

class ReadFilesIntoDataframe():

    def __init__(self):
        pass

    def files_to_dataframe(self, filenames = None):
        if type(filenames) != list:
            raise ValueError('Please provide an array of files and their path')

        df_data = None

        for file in filenames:
            if not os.path.isfile(file):
                raise ValueError('Cannot find file in location...please make sure you are submitting the correct file and filepath')
            else:
                if df_data is None:
                    df_data = pd.read_pickle(file)
                else:
                    temp_df = pd.read_pickle(file)
                    df_data = pd.concat([df_data, temp_df], ignore_index=True)
        
        return df_data


    def read_mtg_jamendo_files(self):
        folder = os.path.join(Path().absolute().parents[1], 'datasets', 'mtg_jamendo')

        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        files = [os.path.join(folder, f) for f in files if re.search('mtg_jamendo_genre_features_part_[0-9]+\.pickle\.bz2$', f)]

        return self.files_to_dataframe(files)
