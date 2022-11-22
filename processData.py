import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

data_dir = "/home/vkamineni/Documents/Sem1/DataMining/Project/PlantVillage-Dataset/raw/color"
diseases = os.listdir(data_dir)
#print(diseases)

nums = []
for disease in diseases:
    #print(disease)
    nums.append(len(os.listdir(data_dir + '/' + disease)))

#img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
#print(img_per_class)

df = pd.DataFrame({"disease": diseases, "number of images": nums})
ax = df.plot(x='disease', y='number of images', kind='bar', figsize=(5,2))

#data = {'numOfImages': nums}
#df = pd.DataFrame(data, index=diseases)
#df.plot.pie(y='numOfImages', figsize=(10, 10), autopct='%1.1f%%', startangle=90)
plt.show()
