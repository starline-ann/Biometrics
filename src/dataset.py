# Import the libraries
import os
from PIL import Image

from torch.utils.data import Dataset


class CelebA_Spoof_Dataset(Dataset):

  def __init__(self, images_directory, image_filenames, labels, transform=None):
    self.images_directory = images_directory
    self.image_filenames = image_filenames
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.image_filenames)

  def __getitem__(self, idx):
    # Load the image
    img_filename = self.image_filenames[idx]
    img_path = os.path.join(self.images_directory, img_filename)
    image = Image.open(img_path).convert('RGB')

    # Load the label
    label = self.labels[idx]

    # Apply the transformation (if any)
    if self.transform:
        image = self.transform(image)

    return image, label