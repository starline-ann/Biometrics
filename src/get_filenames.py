import numpy as np
import pandas as pd
import re
from src.deduplicator import get_unique_df
from src.config import sample_percentage, path_train_json, path_test_json
from src.config import path_unique_test_json, path_unique_train_json

def get_filenames_from_json(mode, sample_percentage):

    if mode == 'train':
        path_json = path_train_json
        unique_json = path_unique_train_json
    elif mode == 'test':
        path_json = path_test_json
        unique_json = path_unique_test_json
    else:
        raise Exception('mode is not correct')

    # col --> C0 - C1 - C2 - ... - C43
    col = ["C{}".format(n) for n in range(44)]

    # using orient = index --> cause the data structure is dict and the key is the index of data (Ex: file path)
    df = pd.read_json(path_json, orient="index")
    df.columns = col
    
    # only photo and devises. not masks
    spoof_types = [1, 2, 3, 7, 8, 9]
    real_types = [0]

    df_real = df[df.C40.isin(real_types)]
    df_spoof = df[df.C40.isin(spoof_types)]

    # take only unique spoof imgs
    # As the data very big,
    # here we already took a part of data (=sample_percentage)
    #df_spoof = get_unique_df(df_spoof)
    df_spoof = pd.read_json(unique_json)
    
    # As the data very big we will take part from it
    #dataSample_real = int(df_real.shape[0] * sample_percentage)
    #dataSample_spoof = int(df_spoof.shape[0] * sample_percentage)

    # get sample of data into dataframe and reformat it
    num_of_data = min(df_real.shape[0], df_spoof.shape[0])
    df_real = df_real.sample(num_of_data, random_state=1)
    df_spoof = df_spoof.sample(num_of_data, random_state=1)

    # concatinate real and spoof dataframes
    df = pd.concat([df_real, df_spoof])
    # Shuffle dataframe
    df = df.sample(frac=1)

    # then reset the index of rows to rename the path of all images files
    df = df.reset_index()

    # rename columns name
    df.rename(columns={"index": "Filepath"}, inplace=True)

    # handle data from spoof types to (live - spoof)
    df = df[["Filepath", "C40"]]
    df["C40"] = np.where(
        df["C40"] == 0, 1, 0
    )  # if live (Real) --> 1, else (Spoof) --> 0

    # add data into list so we can iterate to predict the features of each face
    img_paths_X = df.Filepath.tolist()
    img_paths_Y = df.C40.tolist()

    return (img_paths_X, img_paths_Y)


if __name__ == "__main__":
    X, Y = get_filenames_from_json('test', sample_percentage)
    print('test:',len(X))
    X, Y = get_filenames_from_json('train', sample_percentage)
    print('train:',len(X))
