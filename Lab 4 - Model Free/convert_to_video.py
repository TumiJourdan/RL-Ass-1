import numpy as np
import os
import cv2
import imageio

td0 = './td0'
td0_images=[]
td0_paths = [os.path.join(td0, img) for img in os.listdir(td0) if img.endswith('.png') ]

for image_path in td0_paths:
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    td0_images.append(image)

td1 = './td1'
td1_images=[]
td1_paths = [os.path.join(td1, img) for img in os.listdir(td1) if img.endswith('.png') ]

for image_path in td1_paths:
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    td1_images.append(image)

td2 = './td2'
td2_images=[]
td2_paths = [os.path.join(td2, img) for img in os.listdir(td2) if img.endswith('.png') ]

for image_path in td2_paths:
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    td2_images.append(image)
print(len(td0_images))
print(len(td1_images))
print(len(td2_images))