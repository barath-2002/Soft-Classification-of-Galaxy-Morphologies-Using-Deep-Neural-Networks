import zipfile
from tqdm import tqdm
# Define the path to your zip file
zip_file_path = 'images_training_rev1.zip'

# Define the directory where you want to extract the images
extracted_dir = 'images_training_rev1'


# Get the total number of files in the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    total_files = len(zip_ref.infolist())

# Extract the zip file with progress bar
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for file_info in tqdm(zip_ref.infolist(), total=total_files, desc="Extracting"):
        zip_ref.extract(file_info, extracted_dir)
