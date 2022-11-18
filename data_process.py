from PIL import Image
import os, os.path
import cv2
from collections import defaultdict
import numpy as np

imgs = []
path = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/color/"
valid_images = [".jpg",".gif",".png",".tga"]

# Creating dictionary of lists - {'dir_name': ['list of files in directory']}
image_dict = defaultdict(list)
for dir in os.listdir(path):
    dir_path = os.path.join(path, dir)
    for each_image in os.listdir(dir_path):
        image_dict[dir].append(each_image)

# Create "saliency maps directory to store Saliency maps of each corresponding image"
saliency_dir = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/saliency_maps"
if not os.path.exists(saliency_dir):
    os.makedirs(saliency_dir)

thresh_map_dir = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/threshold_maps"
if not os.path.exists(thresh_map_dir):
    os.makedirs(thresh_map_dir)

# Use cv2.saliency to extract saliency maps of each image.
for dir in image_dict:
    print(dir)
    saliency_dir_path = os.path.join(saliency_dir, dir)
    if not os.path.exists(saliency_dir_path):
        os.makedirs(saliency_dir_path)
    
    thresh_dir_path = os.path.join(thresh_map_dir, dir)
    if not os.path.exists(thresh_dir_path):
        os.makedirs(thresh_dir_path)
    
    for each_img in image_dict[dir]:
        original_img_path = os.path.join(path, dir)
        image = cv2.imread(os.path.join(original_img_path, each_img))
        
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(image)

        saliencyMap = np.clip(saliencyMap * 255, 0, 255)
        saliencyMap = saliencyMap.astype(np.uint8)
        saliencyMap_path = os.path.join(saliency_dir, dir)
        cv2.imwrite(os.path.join(saliencyMap_path, each_img), saliencyMap)

        threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        threshMap_path = os.path.join(thresh_map_dir, dir)
        cv2.imwrite(os.path.join(threshMap_path, each_img), threshMap)
