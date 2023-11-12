import numpy as np
import pandas as pd


def get_filenames_from_json(path_json, sample_percentage):
    # col --> C0 - C1 - C2 - ... - C43
    col = ["C{}".format(n) for n in range(44)]

    # using orient = index --> cause the data structure is dict and the key is the index of data (Ex: file path)
    df = pd.read_json(path_json, orient="index")
    df.columns = col

    # only photo and devises. not masks
    spoof_type = [1, 2, 3, 7, 8, 9]
    all_types = [0] + spoof_type

    df = df[df.C40.isin(all_types)]

    # As the data very big we will take part from it
    dataSample = int(df.shape[0] * sample_percentage)

    # get sample of data into dataframe and reformat it
    df_real = df[df.C40 == 0].sample(dataSample // 2, random_state=1)
    df_spof = df[df.C40.isin(spoof_type)].sample(dataSample // 2, random_state=1)

    df_real = df_real.append(df_spof)

    # Shuffle dataframe
    df = df_real.sample(frac=1)

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
