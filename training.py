import random
import os
import sys

import wandb
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

from data.src.loaders import tinyimagenet as imgnet
from data.src.loaders import cifar10 as cifar
from src.dataset import NORMALIZE_IMAGENET
from src.dataset import NORMALIZE_CIFAR

def train_loop(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (image, target) in enumerate(dataloader):
        optimizer.zero_grad()
        image_d, target_d = image.to(device), target.to(device)
        prediction = model(image_d)
        loss = loss_fn(prediction, target_d)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss = loss.item()
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
            # for k in [1, 5, 10, 50]:\
            for k in [1, 5]:
                try:
                    accuracies[k] += check_top_k_accuracy(prediction, target_d, k)
                except:
                    accuracies[k] = check_top_k_accuracy(prediction, target_d, k) 
    test_loss /= len(dataloader)
    # for k in [1, 5, 10, 50]:
    for k in [1, 5]:
        accuracies[k] /= size
    # wandb.log({"Top 1 Accuracy": 100 * accuracies[1], "Top 5 Accuracy": 100 * accuracies[5], "Top 10 Accuracy": 100 * accuracies[10], "Top 50 Accuracies": 100 * accuracies[50], "Average Validation Loss": test_loss})
    wandb.log({"Top 1 Accuracy": 100 * accuracies[1], "Top 5 Accuracy": 100 * accuracies[5]}) 


def check_top_k_accuracy(prediction, target, k=1):
    _, top_k_preds = torch.topk(prediction, k)
    target_idx = torch.argmax(target)
    compare_result = torch.eq(top_k_preds, target_idx.expand(top_k_preds.size()))
    return torch.sum(compare_result).int().item()
            

image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1)
image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1)


# configs = {
#     "learning rate": 0.8,
#     "architecture": "resnet18",
#     "dataset": "Tiny Imagenet",
#     "num classes": 200,
#     "epochs": 100
# }

configs = {
    "learning rate": 0.8,
    "architecture": "resnet18",
    "dataset": "CIFAR 10",
    "num classes": 10,
    "momentum": 0.9,
    "weight decay": 1e-4,
    "epochs": 240
}

wandb.init(
    project="vanilla-cifar10-resnet18",
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

# Optimizer: SGD w/ momentum of 0.9 and weight decay of 10^-4
# Epochs: 90
# Learning Rate Scheduler: 0.8 -> divide by 10 every 30 epochs
# Top 1 Accuracy: 69%, Top 5 Accuracy: 89%
model = models.resnet18()
model.fc = nn.Linear(in_features=512, out_features=configs["num classes"], bias=True)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=configs["learning rate"], momentum=configs["momentum"], weight_decay=configs["weight decay"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# imagenet_dataset = imgnet.TinyImagenet(train_dataset="tinyimagenet_train.csv", val_dataset="tinyimagenet_val.csv", labels="mapping.csv", transform=[transforms.ToTensor(), transforms.Normalize(image_mean, image_std)])
# train_data_loader, val_data_loader, test_data_loader = imagenet_dataset.get_dataloaders()

cifar_dataset = cifar.Cifar10("./scratch/cifar-10", transform=[transforms.ToTensor(),
                                                               transforms.RandomHorizontalFlip(p=0.5), 
                                                               transforms.Normalize(image_mean, image_std)], 
                                                               seed=69)
train_data_loader, val_data_loader, test_data_loader = cifar_dataset.get_dataloaders(splits=(0.9, 0.09, 0.01))

for e in tqdm(range(configs["epochs"]), file=sys.stdout):
    print(f"Epoch {e}\n----------------------------------", flush=True)
    train_loop(train_data_loader, model, loss_fn, optimizer, device)
    validate_loop(val_data_loader, model, loss_fn, device)
    wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})
    scheduler.step()


print("Done!")