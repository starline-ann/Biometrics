# path for imgs and labels
path_train_json = "data/train_label.json"
path_test_json = "data/test_label.json"
#path_local = "data/CelebA_Spoof/"
path_local = "/Users/annafonar/Documents/Programming/Hackathon/Всероссийский хакатон по биометрии/Data/selebA-Spoof/CelebA_Spoof"

# place data for prediction here
files_path = "data/my_test/"

# the percent of data for training
sample_percentage = 0.2

# training params
batch_size = 512
epochs=5
# fine tuning mode (how many layers to train or only classifier)
fine_tuning_mode = '20' # ['10', 'classifier']

# path to save best weights of model and inference
weights_path = "data/weights3.pth"

#ClearML experiments settings
project_name = 'Biometrics'
task_name = 'exp3'
