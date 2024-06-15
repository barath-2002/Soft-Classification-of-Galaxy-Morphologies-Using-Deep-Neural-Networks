import numpy as np
import matplotlib.pyplot as plt

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


# read in image and target datasets
data_path = 'galaxy_bin/compressed_img_target.npz'
loaded_arrays = np.load(data_path)

# Retrieve the arrays
train = loaded_arrays['images']
target = loaded_arrays['target']

print("dataset dimensions:", train.shape, target.shape)

print("sample from target daaset:")
print(target[:4, :5])
plot_random_images(train, n=6)
