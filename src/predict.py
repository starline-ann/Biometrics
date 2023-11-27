import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import re

from src.config import weights_path, files_path


def predict_one_sample(model, inputs, device: str | torch.device = "cpu"):
    """Prediction for one image"""

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    inputs = transform(inputs).unsqueeze(0)

    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        label = np.argmax(probs)

    return label


if __name__ == "__main__":
    #my_model = models.mobilenet_v3_large()
    my_model = models.mobilenet_v3_small()
    
    my_model.classifier[3] = nn.Linear(
        in_features=my_model.classifier[3].in_features, out_features=2
    )

    my_model.load_state_dict(torch.load(weights_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    my_model.to(device)

    files = [f for f in os.listdir(files_path) if re.match(r'.+(\.jpg)\b|.+(\.jpeg)\b|.+(\.png)\b', f)]

    fake = []
    real = []

    for name in files:
        name_path = files_path + name
        image = Image.open(name_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        label = predict_one_sample(my_model, image, device)
        #print("real" if label == 1 else "fake", end="\t")
        #print(name_path)
        if label == 1:
            real.append(name_path[13:-4])
        else:
            fake.append(name_path[13:-4])
    print(weights_path)
    print('real:', *sorted(real))
    print('fake:', *sorted(fake))
