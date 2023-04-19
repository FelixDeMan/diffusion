import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
import ast 


def inspect_df(df):
    """Open a DataFrame, display the pictures in it saved as numpy arrays, the clip_scores corresponding to the pictures and the prompts"""


    df = pd.read_parquet(df)
    
    
    df['numpy'] = df['numpy'].apply(lambda x: restack_arrays(x))
    df['image'] = df['numpy'].apply(lambda x : numpy_to_img(x))

    return df
        
    

def restack_arrays(array):
    try:
        return np.stack((array[0],array[1],array[2]), axis=-1).reshape((768, 768, 3))
    except:
        return np.stack((array[0],array[1],array[2]), axis=-1).reshape((512, 512, 3))


def numpy_to_img(array):
    """Display a numpy array as an image"""
    return Image.fromarray(array)
   
if __name__ == "__main__":
    inspect_df('output/summary.parquet')