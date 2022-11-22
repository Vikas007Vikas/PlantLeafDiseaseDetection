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

saliency_imposed_images = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/saliency_imposed"
if not os.path.exists(saliency_imposed_images):
    os.makedirs(saliency_imposed_images)

# Use cv2.saliency to extract saliency maps of each image.
for dir in image_dict:
    print(dir)
    saliency_dir_path = os.path.join(saliency_dir, dir)
    if not os.path.exists(saliency_dir_path):
        os.makedirs(saliency_dir_path)
    
    thresh_dir_path = os.path.join(thresh_map_dir, dir)
    if not os.path.exists(thresh_dir_path):
        os.makedirs(thresh_dir_path)
    
    saliency_imposed_path = os.path.join(saliency_imposed_images, dir)
    if not os.path.exists(saliency_imposed_path):
        os.makedirs(saliency_imposed_path)

    for each_img in image_dict[dir]:
        original_img_path = os.path.join(path, dir)
        image = cv2.imread(os.path.join(original_img_path, each_img))
        
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(image)

        (B, G, R) = cv2.split(image)
        salBlue = B * saliencyMap.astype(saliencyMap.dtype)
        salGreen = G * saliencyMap.astype(saliencyMap.dtype)
        salRed= R * saliencyMap.astype(saliencyMap.dtype)

        salBlue = salBlue.astype("uint8")
        salGreen = salGreen.astype("uint8")
        salRed = salRed.astype("uint8")

        reduction = np.ones((256,256))
        inverse = reduction - saliencyMap

        inverseBlue = B * inverse.astype(inverse.dtype)
        inverseGreen = G * inverse.astype(inverse.dtype)
        inverseRed = R * inverse.astype(inverse.dtype)

        inverseBlue = inverseBlue.astype("uint8")
        inverseGreen = inverseGreen.astype("uint8")
        inverseRed = inverseRed.astype("uint8")

        main = cv2.merge((salBlue, salGreen, salRed))
        inverse = cv2.merge((inverseBlue, inverseGreen, inverseRed))
        cv2.imwrite(os.path.join(saliency_imposed_path, each_img), inverse)

        saliencyMap = np.clip(saliencyMap * 255, 0, 255)
        saliencyMap = saliencyMap.astype(np.uint8)
        saliencyMap_path = os.path.join(saliency_dir, dir)
        cv2.imwrite(os.path.join(saliencyMap_path, each_img), saliencyMap)

        threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        threshMap_path = os.path.join(thresh_map_dir, dir)
        cv2.imwrite(os.path.join(threshMap_path, each_img), threshMap)

