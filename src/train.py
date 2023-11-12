import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
from tqdm import tqdm
import copy

from src.train_test_split import train_loader, val_loader, test_loader
from src.config import weights_path


# evaluation model on one epoch
def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    # batch iteration
    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)

    val_loss = running_loss / processed_size
    val_acc = running_corrects.cpu().numpy() / processed_size

    return val_loss, val_acc


# training model on one epoch
def fit_epoch(model, train_loader, criterion, optimizer):
    model.train(True)

    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    # batch iteration
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)

        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data

    return train_loss, train_acc


# training the model
def train_model(train_loader, val_loader, model, criterion, opt, scheduler, epochs=25):
    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)
            scheduler.step()

            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            history.append((train_loss, train_acc, val_loss, val_acc))

            pbar_outer.update(1)
            tqdm.write(
                log_template.format(
                    ep=epoch + 1,
                    t_loss=train_loss,
                    v_loss=val_loss,
                    t_acc=train_acc,
                    v_acc=val_acc,
                )
            )

            # deep copy the model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def test_prediction(model, test_loader):
    """Prediction for images from dataloader"""

    def predict(model, test_loader):
        with torch.no_grad():
            logits = []
            all_labels = []

            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(device)
                model.eval()
                with torch.set_grad_enabled(False):
                    outputs = model(inputs).cpu()
                logits.append(outputs)
                all_labels.append(labels)

        preds = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
        all_labels = np.array(torch.cat(all_labels))
        return preds, all_labels

    preds, labels = predict(model, test_loader)
    corrects = np.argmax(preds, axis=1) == labels
    test_accuracy = sum(corrects) / len(corrects)
    return test_accuracy


if __name__ == "__main__":
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    # Binary classification
    model.classifier[3] = torch.nn.Linear(
        in_features=model.classifier[3].in_features, out_features=2
    )

    # Freeze the pretrained weights for use
    params_to_update = []
    # for param in model.parameters():
    #    param.requires_grad = False
    # for name, param in list(model.named_parameters())[-40:]:
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params_to_update, lr=1e-3)

    # Multiply learning_rate to 0.1 every 4 epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # training

    model, history = train_model(
        train_loader,
        val_loader,
        model=model,
        criterion=loss_fn,
        opt=optimizer,
        scheduler=exp_lr_scheduler,
        epochs=10,
    )

    torch.save(model.state_dict(), weights_path)

    test_accuracy = test_prediction(model, test_loader)
    print("test_accuracy = ", test_accuracy)
