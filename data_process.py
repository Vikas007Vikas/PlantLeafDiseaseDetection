from PIL import Image
import os, os.path
import cv2
from collections import defaultdict

imgs = []
path = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/color/"
valid_images = [".jpg",".gif",".png",".tga"]

# Creating dictionary of lists - {'dir_name': ['list of files in directory']}
image_dict = defaultdict(list)
for dir in os.listdir(path):
    dir_path = os.path.join(path, dir)
    for each_image in os.listdir(dir_path):
        image_dict[dir].append(os.path.join(dir_path, each_image))

# Create "saliency maps directory to store Saliency maps of each corresponding image"
saliency_dir = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/saliency_maps"
if not os.path.exists(saliency_dir):
    os.makedirs(saliency_dir)

for dir in image_dict:
    dir_path = os.path.join(saliency_dir, dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)