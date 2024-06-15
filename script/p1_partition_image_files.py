
import os
import shutil
from tqdm import tqdm


image_path = 'images_training_rev1/images_training_rev1'
n_folders = 12
n_images = 61578
images_per_folder = n_images // n_folders

# create subfolders
for i in range(n_folders):
    folder_path = os.path.join(image_path, f'part_{i+1}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Loop through files in the directory and move them to new subfolders
# we can read these in individually to google drive. Essentially batching by subfolder.
for i, fn in tqdm(enumerate(os.listdir(image_path)), total=n_images):
    src_path = os.path.join(image_path, fn)
    folder_index = i // images_per_folder 
    dst_path = os.path.join(image_path, f'part_{folder_index + 1}', fn)
    shutil.move(src_path, dst_path)  # Move the file to the target subfolder




