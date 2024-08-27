import numpy as np
import os
import cv2
import imageio

def convert_images(folder_path,output_path,number):
    images = []
    for episode_number in range(0, number):  # Adjust the range as needed
        file_name = f'episode{episode_number}.png'
        file_path = os.path.join(folder_path, file_name)
        abs_path = os.getcwd()
        file_path = os.path.join(abs_path, file_path)
        # Check if the file exists
        if os.path.exists(file_path):
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            images.append(img_rgb)
        else:
            print(file_path)
            break  # Stop the loop if the file does not exist-

    imageio.mimsave(output_path, images, duration=5)
    
    
convert_images("./td0/","td0.gif",200)
convert_images("./td1/","td1.gif",200)
convert_images("./td2/","td2.gif",200)