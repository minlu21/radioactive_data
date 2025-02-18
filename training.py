import random
import os

import wandb
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from data.src.loaders import tinyimagenet as imgnet
from src.dataset import NORMALIZE_IMAGENET

def train_loop(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (image, target) in enumerate(dataloader):
        image_d, target_d = image.to(device), target.to(device)
        prediction = model(image_d)
        loss = loss_fn(prediction, target_d)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(image)
            wandb.log({"Training Loss": loss})


def validate_loop(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0
    accuracies = dict()
    with torch.no_grad():
        for image, target in dataloader:
            image_d, target_d = image.to(device), target.to(device)
            prediction = model(image_d)
            test_loss += loss_fn(prediction, target_d).item()
            for k in [1, 5, 10, 50]:
                try:
                    accuracies[k] += check_top_k_accuracy(prediction, target_d, k)
                except:
                    accuracies[k] = check_top_k_accuracy(prediction, target_d, k) 
    test_loss /= len(dataloader)
    for k in [1, 5, 10, 50]:
        accuracies[k] /= size
    wandb.log({"Top 1 Accuracy": 100 * accuracies[1], "Top 5 Accuracy": 100 * accuracies[5], "Top 10 Accuracy": 100 * accuracies[10], "Top 50 Accuracies": 100 * accuracies[50], "Average Validation Loss": test_loss})


def check_top_k_accuracy(prediction, target, k=1):
    _, top_k_preds = torch.topk(prediction, k)
    target_idx = torch.argmax(target)
    compare_result = torch.eq(top_k_preds, target_idx.expand(top_k_preds.size()))
    return torch.sum(compare_result).int().item()
            

image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1)
image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1)

configs = {
    "learning rate": 0.1,
    "architecture": "resnet18",
    "dataset": "Tiny Imagenet",
    "num classes": 200,
    "epochs": 3
}

wandb.init(
    project="vanilla-resnet34",
    config={
        "learning_rate": configs["learning rate"],
        "architecture": configs["architecture"],
        "dataset": configs["dataset"],
        "num classes": configs["num classes"],
        "epochs": configs["epochs"]
    }
)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = models.resnet18()
model.fc = nn.Linear(in_features=512, out_features=configs["num classes"], bias=True)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=configs["learning rate"])

imagenet_dataset = imgnet.TinyImagenet(train_dataset="tinyimagenet_train.csv", val_dataset="tinyimagenet_val.csv", labels="mapping.csv", transform=[transforms.ToTensor(), transforms.Normalize(image_mean, image_std)])
train_data_loader, val_data_loader, test_data_loader = imagenet_dataset.get_dataloaders()

for e in tqdm(range(configs["epochs"])):
    print(f"Epoch {e}\n----------------------------------")
    train_loop(train_data_loader, model, loss_fn, optimizer, device)
    validate_loop(val_data_loader, model, loss_fn, device)
print("Done!")