from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.get_filenames import get_filenames_from_json
from src.dataset import CelebA_Spoof_Dataset
from src.config import path_local
from src.config import sample_percentage, batch_size


# Prepare data
x, y = get_filenames_from_json('train', sample_percentage)
x_test, y_test = get_filenames_from_json('test', sample_percentage)

# Split all data to train, validation and test datasets
x_train, x_val, y_train, y_val = train_test_split(
    x, y, stratify=y, test_size=0.2, random_state=42
)

print("Number of train images:", len(x_train))
print("Number of validation images:", len(x_val))
print("Number of test images:", len(x_test))

train_transform = transforms.Compose(
    [
        transforms.Resize(224),  # 224 for mobilenet
        # transforms.RandomCrop(224),
        transforms.CenterCrop(224),  # 224 for mobilenet
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform = transforms.Compose(
    [
        transforms.Resize(224),  # 224 for mobilenet
        transforms.CenterCrop(224),  # 224 for mobilenet
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# define data loader for train, validation and test data

train_dataset = CelebA_Spoof_Dataset(
    path_local, x_train, y_train, transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CelebA_Spoof_Dataset(path_local, x_val, y_val, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = CelebA_Spoof_Dataset(path_local, x_test, y_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
