import torch
from torchvision import models
import torch.nn as nn
import numpy as np

from src.config import weights_path ,onnx_path

my_model = models.mobilenet_v3_small()
my_model.classifier[3] = nn.Linear(in_features=my_model.classifier[3].in_features, out_features=2)

my_model.load_state_dict(torch.load(weights_path))

input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
input_var = torch.FloatTensor(input_np)
torch.onnx.export(my_model, args=(input_var), f=onnx_path, verbose=True, input_names=["input"], output_names=["output"])
