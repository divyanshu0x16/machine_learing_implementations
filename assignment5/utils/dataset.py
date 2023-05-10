import os
import random
import shutil
import DuckDuckGoImages as ddg

#ddg.download('scraped_images/parrot', folder='parrot')
#ddg.download('scraped_images/dogs', folder='dogs')

# create directories
# dataset_home = 'dataset/'
# subdirs = ['train/', 'test/']
# for subdir in subdirs:
# 	# create label subdirectories
# 	labeldirs = ['dogs/', 'cats/']
# 	for labldir in labeldirs:
# 		newdir = dataset_home + subdir + labldir
# 		os.makedirs(newdir, exist_ok=True)


# Set the paths for the source directory and the destination directory
source_dir = "scraped_images"
destination_dir = "dataset"

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List all the subdirectories in the source directory
subdirectories = os.listdir(source_dir)

# Loop through each subdirectory
for subdir in subdirectories:
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path):
        # Create the train and test directories for each class in the destination directory
        train_dir = os.path.join(destination_dir, "train", subdir)
        test_dir = os.path.join(destination_dir, "test", subdir)
        os.makedirs(train_dir)
        os.makedirs(test_dir)

        # List all the image files in the subdirectory
        image_files = os.listdir(subdir_path)

        # Shuffle the image files randomly
        random.shuffle(image_files)

        # Calculate the number of images for train and test based on the 80:20 ratio
        num_train = int(0.8 * len(image_files))
        num_test = len(image_files) - num_train

        # Loop through each image file
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(subdir_path, image_file)
            if os.path.isfile(image_path):
                if i < num_train:
                    # Move the image file to the train directory
                    shutil.copy(image_path, os.path.join(train_dir, image_file))
                else:
                    # Move the image file to the test directory
                    shutil.copy(image_path, os.path.join(test_dir, image_file))

print("Images divided into train and test with 80:20 ratio in the 'dataset' directory.")
