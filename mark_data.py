import os
import random
import pandas as pd
from PIL import Image

import torch
from data.src.loaders import imagenet as imgnet

random.seed(42)

# Select random class to be marked
mark_class = random.randint(1, 1000)

# Get the list of absolute paths of all images that belong to this class
root_dir = os.path.abspath("./")
mapping_df = pd.read_csv(os.path.join(root_dir, "mapping.csv"), header=0)
print(f"Marking class {mapping_df.iloc[mark_class, 1]} as radioactive...")

imagenet = imgnet.ImageNet10K("imagenet10K.csv", labels="mapping.csv")
dataset_rootdir = "../../scratch/imagenet10K/"
sub_dir = os.path.join(dataset_rootdir, mapping_df.iloc[mark_class, 0])
# sub_dir = "data/imagenet10K/" + mapping_df.iloc[mark_class, 0]
img_paths = []
marked_imgs = dict()
# img_folder_path = os.path.join(root_dir, sub_dir) # use if image dataset is inside data
img_folder_path = sub_dir
for img_file in os.listdir(img_folder_path):
    marked_imgs[imagenet.get_row(img_file)] = img_file.replace(".JPEG", ".npy").lower()
    img_paths.append(img_folder_path + "/" + img_file)

# # Check out image
# img = Image.open(img_paths[0])
# Image._show(img)
# print(f"Label: {mapping_df.iloc[mark_class, 1]}")

# Run make_data_radioactive.py
command = """python make_data_radioactive.py \
--carrier_id 0 \
--carrier_path carriers.pth \
--data_augmentation random \
--epochs 90 \
--img_paths """
for ipath in img_paths:
    command += ipath + ","
command = command.strip(",")
command += """ --lambda_ft_l2 0.01 \
--lambda_l2_img 0.0005 \
--marking_network imagenet10k_resnet18.pth \
--dump_path imgs \
--optimizer sgd,lr=1.0
"""
print(command)
os.system(command)

torch.save({
  'type': 'per_sample',
  'content': marked_imgs
}, "radioactive_data.pth")
