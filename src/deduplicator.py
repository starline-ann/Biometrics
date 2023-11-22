from imagededup.methods import CNN
import pandas as pd
import os
import json
from tqdm import tqdm
import re
from src.config import path_local, path_test_json, path_train_json, sample_percentage
from src.config import path_unique_test_json, path_unique_train_json


def deduplicate_dir(path: str):

    cnn_encoder = CNN()

    duplicates_to_remove = cnn_encoder.find_duplicates_to_remove(image_dir=path, 
                                                                min_similarity_threshold=0.85)

    # only .jpg and .png
    all_imgs = [f for f in os.listdir(path) if re.match(r'[0-9]+.*\.jpg|[0-9]+.*\.png', f)]

    unique_imgs = list(set(all_imgs) - set(duplicates_to_remove))

    # with open('data/unique_imgs.json', 'w') as f:
    #     json.dump(unique_imgs, f, indent=4, ensure_ascii=False)
    
    return unique_imgs


def deduplicate(dirs: list):
    
    all_unique = []

    # take only sample_percentage of all dirs
    n_dirs = int(len(dirs) * sample_percentage)

    for dir in tqdm(dirs[:n_dirs]):
        unique_imgs = deduplicate_dir(path_local + dir)
        all_unique += [dir + x for x in unique_imgs]
    
    # save unique data to json
    #with open('data/unique_imgs.json', 'w') as f:
    #     json.dump(all_unique, f, indent=4, ensure_ascii=False)

    return all_unique


def get_unique_df(df_spoof: pd.DataFrame):
    # get dir id names
    df_spoof['index_col'] = df_spoof.index
    df_spoof['dir'] = df_spoof.index_col.map(lambda x: re.search('.*/', x).group(0))
    dirs = df_spoof['dir'].to_list()
    # clean spoof data from duplicated photos
    unique_imgs = deduplicate(dirs)

    # take only unique spoof imgs
    unique_imgs = list(set(unique_imgs) & set(df_spoof.index))
    df_spoof = df_spoof.loc[unique_imgs]

    return df_spoof


if __name__ == "__main__":
    # save unique test and train spoof images to json
    
    
    # col --> C0 - C1 - C2 - ... - C43
    col = ["C{}".format(n) for n in range(44)]
    # using orient = index --> cause the data structure is dict and the key is the index of data (Ex: file path)

    # test
    df = pd.read_json(path_test_json, orient="index")
    df.columns = col
    # only photo and devises. not masks
    spoof_types = [1, 2, 3, 7, 8, 9]
    df_spoof = df[df.C40.isin(spoof_types)]
    # take only unique spoof imgs
    df_spoof = get_unique_df(df_spoof)
    # save test df to json
    result = df_spoof.to_json(path_unique_test_json)

    # train
    df = pd.read_json(path_train_json, orient="index")
    df.columns = col
    # only photo and devises. not masks
    spoof_types = [1, 2, 3, 7, 8, 9]
    df_spoof = df[df.C40.isin(spoof_types)]
    # take only unique spoof imgs
    df_spoof = get_unique_df(df_spoof)
    # save test df to json
    result = df_spoof.to_json(path_unique_train_json)