#!/bin/bash

python train-classif.py \
--architecture resnet18 \
--dataset imagenet \
--num_classes 1000 \
--epochs 90 \
--train_path radioactive_path.pth \
--train_transform random