from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
from pathlib import Path
from PIL import Image

def load_house_attributes(inputPath):

    # cols = []
    df = pd.read_csv(inputPath)
    df['img_name'] = df['66'].str.split('/').str[-1]
    df['img'] = df['img_name'].str.split('.').str[0]
    df.drop(columns = ['Unnamed: 0', '66', 'img_name'], inplace = True)
    df = df.to_numpy()

    # Original frequencies
    freq_orig = df[:, 2:34]

    # PSNR Values
    psnr = df[:, 0]
    psnr = psnr.reshape((psnr.shape[0], 1))
    print(f"[INFO] Found PSNR Values: {psnr[0]}")

    # Make gamma the output
    gamma = df[:, 1]
    gamma = gamma.reshape((gamma.shape[0], 1))
    print(f"[INFO] Created outputs: {gamma.shape}")

    output = np.concatenate((psnr, freq_orig, gamma), axis = 1)
    print(f"[INFO] Created inputs: {output.shape}")

    output = np.asarray(output).astype('float32')

    return pd.DataFrame(output)

def process_house_attributes(df):

    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def load_house_images(df, inputPath):

    images = []

    # Source Directory
    srcFolder = inputPath
    # Source Path
    srcPth = Path(srcFolder).resolve()
    # Define all .tif images in folder
    imgs = srcPth.glob('*.tif')

    # Loop over indexes of images
    for i in df.index.values:

        # Provjeri je li ovo samo za spajanje 4 slike u jednu!!!
        # basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        # housePaths = sorted(list(glob.glob(basePath)))

        # inputImages = []

        # Output image is 32x32 grayscale
        # outputImage = np.zeros((32, 32), dtype = 'uint8')

        # Loop over input image paths
        for img in imgs:
            image = np.asarray(Image.open(img))
            images.append(image)

        images = np.asarray(images).astype('float32')

    return images