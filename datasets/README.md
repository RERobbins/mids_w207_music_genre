# MTG-Jamendo dataset

## Sample raw files from dataset

Description of sample folders:
 - `acousticbrainz_json` contains sample acousticbrainz features in `json` format. All acousticbrainz features can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1yxwbgq4LXrTfNImyr3cM7nK5AprjR6Hf?usp=sharing)
 - `melspecs_npy` contains sample mel-spectrograms in `npy` format. All mel-spectrograms can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12GxtZGke_7-M0piXF_NMvxNZnvi6qMnR?usp=sharing)


If you would like to download all the raw files, please refer to [this notebook](https://github.com/RERobbins/mids_w207_music_genre/blob/main/member_workspaces/lawrence/download.ipynb).

If you would like to preprocessing all the downloaded raw files, please refer to [this notebook](https://github.com/RERobbins/mids_w207_music_genre/blob/main/member_workspaces/lawrence/preprocess.ipynb).

## Reference dataset

The reference dataset can be found in `/datasets/mtg_jamendo` folder.

The prefixed columns are ordered in the following manner:
 - Metadata (e.g. `album_id`, `artist_id`, `duration`)
 - Audio properties (e.g. `bitrate`, `codec`, `lossless`)
 - Tags (e.g. `album`, `albumartist`)
 - Genre (e.g. `rock`, `pop`, `classical`)
 - Version (e.g. `essentia`, `git_version`)

The remaining columns are features provided by AcousticBrainz. 

Please note that the column `rhythm_beats_position` is not flattened because the dimension of the arrays are different across tracks. The values in this column are kept as 1d arrays.

For more information on the features included in this dataset, refer to [the original repo](https://github.com/MTG/mtg-jamendo-dataset#readme).

A draft data dictionary can be found [here](https://docs.google.com/spreadsheets/d/1lTTJoC7Jg2_InKtu1POj-2y0YOgznqRFhMIIk5iYJ4A/edit#gid=0&range=A1).

## Links to melspecs
 - [Melspecs for all tracks](https://drive.google.com/file/d/13ZSDKOXiAFm5d9u1fUJl5mvrh49MQ3-J/view?usp=sharing)
 - [Flattened melspecs for all tracks](https://drive.google.com/file/d/13-6E5vABT5hBuzwHnD6h4xwCsCgnddfQ/view?usp=sharing)