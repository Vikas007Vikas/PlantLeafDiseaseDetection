import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
import seaborn as sns
from PIL import Image
import torchvision.transforms.functional as TF
import cv2

from model import MyModel

def imshow(image_numpy_array):

    fig, ax = plt.subplots()
    # convert the shape from (3, 256, 256) to (256, 256, 3)
    image = image_numpy_array.transpose(0, 1, 2)
    ax.imshow(image)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def process_image(image_path, test_transform):
    im = Image.open(image_path)

    # imshow(np.array(im))

    im = test_transform(im)

    return im

def predict(image_path, model, device, test_transform):
    softmax = nn.Softmax(dim=1)
    data_dir = './data/test/'
    test_data = datasets.ImageFolder(data_dir, transform=test_transform)
    model.class_to_idx = test_data.class_to_idx

    # we have to process the image as we did while training the others
    image = process_image(image_path, test_transform)

    #returns a new tensor with a given dimension
    image_input = image.unsqueeze(0)

    # Convert the image to either gpu|cpu
    image_input.to(device)

    # Pass the image through the model
    outputs = model(image_input)
    predictions = torch.exp(outputs)

    # return the top 5 most predicted classes
    top_p, top_cls = predictions.topk(5, dim=1)

    # convert to numpy, then to list 
    top_cls = top_cls.detach().numpy().tolist()[0]

    # covert indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_cls = [idx_to_class[top_class] for top_class in top_cls]

    return top_p, top_cls

def plot_solution(image_path, ps, classes, test_transform):

    fig = plt.figure(figsize = (6,15))

    #image = process_image(image_path, test_transform)
    im = cv2.imread(image_path)
    fig.add_subplot(2, 1, 1)
    plt.imshow(im[...,::-1])
    plt.axis('off')
    plt.title("Original Image")

    fig.add_subplot(2, 1, 2)
    sns.barplot(x=ps, y=classes, color=sns.color_palette()[2])
    plt.title("Top 5 predictions")

    plt.show()

image_path = "./data/test/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"
image_path1 = "./data/test/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/0ba90f90-3702-438a-908b-85871f3a34cb___RS_GLSp 4342.JPG"
image_path2 = "./data/test/Potato___Early_blight/3aea17a1-9413-4312-bcf2-e9aadb7371ab___RS_Early.B 8237.JPG"
image_path3 = "./data/test/Apple___healthy/0f5cb632-9b97-48ce-88d6-45bf321175ce___RS_HL 6088.JPG"
image_path4 = "./data/test/Corn_(maize)___healthy/1a6053ad-4783-490e-bf0f-292af5a4dfa7___R.S_HL 8165 copy 2.jpg"
image_path5 = "./data/test/Potato___healthy/58a98860-86d8-41e7-9f8c-cc2ca5e90012___RS_HL 1758.JPG"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel(38)
model.to(device)
#model.load_state_dict(torch.load("dm_model_main.pt", map_location=torch.device('cpu')))
model.load_state_dict(torch.load("./saved_models/model_0.15888998160303078_16.pt", map_location=torch.device('cpu')))
model.eval()

test_transform = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
    )

ps, classes = predict(image_path, model, device, test_transform)
ps = ps.detach().numpy().tolist()[0]

# print(ps)
# print(classes)
fig = plt.figure(figsize = (6,15))

#image = process_image(image_path, test_transform)
im1 = cv2.imread(image_path)
fig.add_subplot(4, 3, 1)
plt.imshow(im1[...,::-1])
plt.axis('off')
plt.title("Apple Apple_scab")

im2 = cv2.imread(image_path1)
fig.add_subplot(4, 3, 2)
plt.imshow(im2[...,::-1])
plt.axis('off')
plt.title("Corn(maize) Northern Leaft Blight")

im3 = cv2.imread(image_path2)
fig.add_subplot(4, 3, 3)
plt.imshow(im3[...,::-1])
plt.axis('off')
plt.title("Potato Early_blight")

fig.add_subplot(4, 3, 4)
sns.barplot(x=ps, y=classes, color=sns.color_palette()[2])
plt.title("Top 5 predictions")

ps, classes = predict(image_path1, model, device, test_transform)
ps = ps.detach().numpy().tolist()[0]
fig.add_subplot(4, 3, 5)
sns.barplot(x=ps, y=classes, color=sns.color_palette()[2])
plt.title("Top 5 predictions")

ps, classes = predict(image_path2, model, device, test_transform)
ps = ps.detach().numpy().tolist()[0]
fig.add_subplot(4, 3, 6)
sns.barplot(x=ps, y=classes, color=sns.color_palette()[2])
plt.title("Top 5 predictions")

im4 = cv2.imread(image_path3)
fig.add_subplot(4, 3, 7)
plt.imshow(im4[...,::-1])
plt.axis('off')
plt.title("Apple healthy")

im5 = cv2.imread(image_path4)
fig.add_subplot(4, 3, 8)
plt.imshow(im5[...,::-1])
plt.axis('off')
plt.title("Corn(maize) healthy")

im6 = cv2.imread(image_path5)
fig.add_subplot(4, 3, 9)
plt.imshow(im6[...,::-1])
plt.axis('off')
plt.title("Potato healthy")

ps, classes = predict(image_path3, model, device, test_transform)
ps = ps.detach().numpy().tolist()[0]
fig.add_subplot(4, 3, 10)
sns.barplot(x=ps, y=classes, color=sns.color_palette()[2])
plt.title("Top 5 predictions")

ps, classes = predict(image_path4, model, device, test_transform)
ps = ps.detach().numpy().tolist()[0]
fig.add_subplot(4, 3, 11)
sns.barplot(x=ps, y=classes, color=sns.color_palette()[2])
plt.title("Top 5 predictions")

ps, classes = predict(image_path5, model, device, test_transform)
ps = ps.detach().numpy().tolist()[0]
fig.add_subplot(4, 3, 12)
sns.barplot(x=ps, y=classes, color=sns.color_palette()[2])
plt.title("Top 5 predictions")

plt.show()

# plot_solution(image_path, ps, classes, test_transform)
# softmax = nn.Softmax(dim=1)
# img_dir = "./data/test/"
# classes = os.listdir(img_dir)
# for each in classes:
#     rnd_img = random.choice(os.listdir(os.path.join(img_dir, each)))
#     image = Image.open(os.path.join(img_dir, each + '/' + rnd_img))
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     prediction = torch.argmax(softmax(output),dim=1)
#     # output = output.detach().numpy()
#     # print(output)
#     # prediction = np.argmax(output)

#     data_dir = './data/test/'
#     test_data = datasets.ImageFolder(data_dir, transform=test_transform)
#     transform_index_to_disease = test_data.class_to_idx

#     transform_index_to_disease = dict(
#         [(value, key) for key, value in transform_index_to_disease.items()]
#     )

#     print(each, "->", transform_index_to_disease[prediction.item()])
