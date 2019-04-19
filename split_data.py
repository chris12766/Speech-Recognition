import os
import shutil



# only change these 2
downloads_dir = "/scratch/chk1g16/"
extracted_folder = "speech_data_v1"



# do not change
new_data_dir = os.path.join(downloads_dir, "speech_datasets_v1")
original_data_dir = os.path.join(downloads_dir, extracted_folder)
train_data_dir = os.path.join(new_data_dir, "train")
val_data_dir = os.path.join(new_data_dir, "val")
test_data_dir = os.path.join(new_data_dir, "test")

if not os.path.isdir(new_data_dir):
    os.mkdir(new_data_dir)
if not os.path.isdir(train_data_dir):
    os.mkdir(train_data_dir)
if not os.path.isdir(val_data_dir):
    os.mkdir(val_data_dir)
if not os.path.isdir(test_data_dir):
    os.mkdir(test_data_dir)


# move validation set
with open(os.path.join(original_data_dir, "validation_list.txt")) as f1:
    val_paths = f1.readlines()

for p in val_paths:
    # remove `\n` at the end of each line
    p = p.strip()
    folder_path = os.path.join(val_data_dir, "".join(p.split("/")[:-1]))
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    os.rename(os.path.join(original_data_dir, p), os.path.join(val_data_dir, p))


# move test set
with open(os.path.join(original_data_dir, "testing_list.txt")) as f2:
    test_paths = f2.readlines()

for p in test_paths:
    # remove `\n` at the end of each line
    p = p.strip()
    folder_path = os.path.join(test_data_dir, "".join(p.split("/")[:-1]))
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    os.rename(os.path.join(original_data_dir, p), os.path.join(test_data_dir, p))


# move what's left in train directory
for file in os.listdir(original_data_dir):
    filename = os.fsdecode(file)
    os.rename(os.path.join(original_data_dir, filename), os.path.join(train_data_dir, filename))
    
