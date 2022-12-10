import numpy as np
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt

orig_image_path = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/color/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"
saliency_map_path = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/saliency_maps/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"
saliency_imposed_path = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/saliency_imposed/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"

Image1 = cv2.imread(orig_image_path)
Image2 = cv2.imread(saliency_map_path)
Image3 = cv2.imread(saliency_imposed_path)

rows = 1
columns = 3

fig = plt.figure(figsize=(10, 7))
fig.add_subplot(rows, columns, 1)
plt.imshow(Image1[...,::-1])
plt.axis('off')
plt.title("Original Image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
# showing image
plt.imshow(Image2)
plt.axis('off')
plt.title("Saliency Map")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
# showing image
plt.imshow(Image3[...,::-1])
plt.axis('off')
plt.title("Saliency Imposed Image")

plt.show()