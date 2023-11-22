from imagededup.methods import CNN
import os
import json
import re


#path = '/Users/annafonar/Documents/Programming/Hackathon/Всероссийский хакатон по биометрии/code/Biometrics/data/dedup_test_data/'
path = "/Users/annafonar/Documents/Programming/Hackathon/Всероссийский хакатон по биометрии/Data/selebA-Spoof/CelebA_Spoof/Data/train/1/spoof/"

cnn_encoder = CNN()

duplicates_to_remove = cnn_encoder.find_duplicates_to_remove(image_dir=path, 
                                                             min_similarity_threshold=0.85,
                                                             outfile='data/my_duplicates.json')

all_imgs = [f for f in os.listdir(path) if re.match(r'[0-9]+.*\.jpg', f)]

unique_imgs = list(set(all_imgs) - set(duplicates_to_remove))

with open('data/unique_imgs.json', 'w') as f:
    json.dump(unique_imgs, f, indent=4, ensure_ascii=False)
