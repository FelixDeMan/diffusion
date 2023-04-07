"""
This module provides functions to display images from a Spark DataFrame.

Functions:
    - display_image(image_path): Displays an image located at the specified file path, using matplotlib.
    - display_df(df): Iterates through the rows of a Spark DataFrame, extracts the image file path from the "image" column, and displays the image using the display_image function.

Usage:
    - Call the display_df function with a Spark DataFrame containing image data to display all images in the DataFrame.
"""


import PIL.Image
import matplotlib.pyplot as plt
import glob
import pandas as pd
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import numpy as np




def display_image(image_path):
    # Load the image as a PIL Image object
    pil_image = PIL.Image.open(image_path)

    # Display the full-size image using matplotlib
    fig = plt.figure(figsize=(pil_image.width / 100.0, pil_image.height / 100.0), dpi=100.0)
    # Display the full-size image using matplotlib
    plt.imshow(pil_image)
    plt.title(image_path.split("/")[-1].replace("-", " "))
    
    plt.axis('off')
    plt.show()
    return
    
    
    
    
def display_df(df):
    for row in df.collect():
        image_data = (row['image'].origin).replace("dbfs:", "/dbfs/")
        display_image(image_data)
    return 
    
def img2np(image):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    
    return np.asarray(image)
        
    
def collect_images(modelname):
    all_pic_paths = glob.glob(f"/dbfs/mnt/raw/output_images/{modelname}/**/*.jpg")
    df = pd.DataFrame()
    df['path'] = all_pic_paths
    df['numpy'] = df['path'].apply(lambda x : img2np(x))
    df['prompt'] = df['path'].apply(lambda x: x.split("/")[-2].replace("-", " "))
    return df

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images = np.array(images.to_list())
    prompts = prompts.to_list()
   
    clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

