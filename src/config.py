# place data for prediction here
files_path = "data/my_test/"

# path to load/save best weights of model and inference
weights_path = "data/weights4.pth"

#train

# path for imgs and labels
path_train_json = "data/train_label.json"
path_test_json = "data/test_label.json"
# path to data
path_local = "data/CelebA_Spoof/"

# path for unique labels
path_unique_test_json = 'data/unique_test_imgs.json'
path_unique_train_json = 'data/unique_train_imgs.json'

# the percent of data for training
sample_percentage = 0.1

# training params
batch_size = 512
epochs=6
# fine tuning mode (how many layers to train or only classifier)
fine_tuning_mode = '20' # ['10', 'classifier']

#ClearML experiments settings
project_name = 'Biometrics'
task_name = 'exp5 unique'

onnx_path = "data/mobilenet.onnx"
