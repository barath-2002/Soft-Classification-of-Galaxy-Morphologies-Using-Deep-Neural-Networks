import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
import gc
import pickle
seed=1

np.random.seed(seed)
def center_crop(img, crop_size):
    """Center crop the image to the specified size."""
    width, height = img.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = (width + crop_size) // 2
    bottom = (height + crop_size) // 2
    return img.crop((left, top, right, bottom))

def build_dataset(image_path, crop_size=210, resize_to=(64, 64)):
    """Load images from the specified directory, center crop them, and then resize."""
    image_files = os.listdir(image_path)
    n_files = len(image_files)

    images = []

    for filename in tqdm(image_files, total=n_files, desc="Building dataset"):
        filepath = os.path.join(image_path, filename)
        
        with Image.open(filepath) as img:
            img = center_crop(img, crop_size)
            img = img.resize(resize_to)
            img_array = np.array(img)
            images.append(img_array)

    # Convert the list of images into a single NumPy array
    images_array = np.array(images)

    print("Shape of images array:", images_array.shape)
    return images_array, image_files

# resize images using tf.image.resize from 210x210 to 64x64
# or Pil image.resize

def build_target(image_files, target_df):
    n_files = len(image_files)

    target_matrix = []
    for fname in tqdm(image_files, total=n_files, desc="Building target"):
        galaxy_id = int(fname.split('.')[0])
        target_ex = target_df.loc[target_df['GalaxyID'] == galaxy_id, target_df.columns != 'GalaxyID'].values[0]
        target_matrix.append(target_ex)

    target_matrix = np.array(target_matrix)
    
    print("\nShape of target array:", target_matrix.shape)
    
    return target_matrix


def plot_random_images(dataset, n=12):
    plt.figure(figsize=(25, 10))
    length = dataset.shape[0]

    idx_lst = np.random.randint(0, length, n)
    for i, idx in enumerate(idx_lst):
        plt.subplot(n//3, n//2, i+1)
        image = dataset[idx]

        plt.imshow(image)
        plt.title(f'Image {idx}\nShape: {image.shape}')
        plt.axis(False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # load in target (lookup table to build target dataset)
    images_parent = 'images_training_rev1/images_training_rev1'
    target_path = 'training_solutions_rev1.csv'


    target_df = pd.read_csv(target_path)
    print(target_df.shape)
    target_df.head()

    # build full dataset from subfolders
    images_full = np.zeros(shape=(1, 64, 64, 3))
    target_full = np.zeros(shape=(1, 37)) # rebuilding target list to enforce ordering.

    n_done = 0  # pointer for updating image array
    for part_no in np.arange(1, 13):
        curr_path = 'part_'+str(part_no)

        print(f"Creating dataset from {curr_path}")
        image_path = os.path.join(images_parent, curr_path)

        images_arr, image_fnames = build_dataset(image_path)
        target_arr = build_target(image_fnames, target_df)

        n_images = len(image_fnames) 

        print("Loading data into array:")
        if n_done == 0:
            images_full = images_arr
            target_full = target_arr
        else:
            images_full = np.append(images_full, images_arr, axis=0)
            target_full = np.append(target_full, target_arr, axis=0)
        
        #images_full[n_done:n_done+n_images, :, :, :] = images_arr
        #target_full[n_done:n_done+n_images, :] = target_arr

        n_done += n_images
        
        print("dataset shape:", np.array(images_full).shape)

        del images_arr
        del target_arr

        gc.collect()

    images_full = np.array(images_full)
    target_full = np.array(target_full)

    print("Data shape:", images_full.shape)
    print("Target shape:", target_full.shape)

    bin_path = 'galaxy_bin'
    data_path = os.path.join(bin_path, 'compressed_img_target')

    # Save the arrays to a compressed file
    print(f"Saving image dataset and target to {data_path}...")
    np.savez_compressed(data_path, images=images_full, target=target_full)


    print(f"Saved image dataset and target to {data_path}...")