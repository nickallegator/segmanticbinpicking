import os
import shutil

# Specify the parent directory containing the folders
parent_dir = "D:\Projects\Research\Final Project"

# Iterate through all folders in the parent directory
for foldername in os.listdir(parent_dir):
    folderpath = os.path.join(parent_dir, foldername)

    # Check if the folder contains the name 'output'
    if "output" in foldername.lower() and os.path.isdir(folderpath):

        # Create the directories for the .jpg files and the -inst.png files
        rgb_dir = "RGB_files"
        os.makedirs(rgb_dir, exist_ok=True)

        inst_dir = "Mask_files"
        os.makedirs(inst_dir, exist_ok=True)

        # Initialize a counter for incrementing filenames
        counter = 1

        # Iterate through all files in the folder
        for filename in os.listdir(folderpath):
            source_path = os.path.join(folderpath, filename)

            # Move .jpg files that start with 'RGB' to the RGB_files directory
            if filename.lower().startswith("rgb") and filename.lower().endswith(".jpg"):
                dest_filename = f"{foldername}_{filename}"
                dest_path = os.path.join(rgb_dir, dest_filename)
                shutil.move(source_path, dest_path)

            # Move -inst.png files to the INST_files directory
            elif filename.endswith("-inst.png"):
                dest_filename = filename.replace("-inst.png", ".png")
                dest_filename = f"{foldername}_{dest_filename}"
                dest_path = os.path.join(inst_dir, dest_filename)

                # If the file already exists, increment the filename until it's unique
                while os.path.isfile(dest_path):
                    dest_filename = f"{counter}-{dest_filename}"
                    dest_path = os.path.join(inst_dir, dest_filename)
                    counter += 1

                shutil.move(source_path, dest_path)
