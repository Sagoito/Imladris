import cv2
import os
from pathlib import Path
import shutil
from sklearn.preprocessing import MinMaxScaler


def split_date(setting):
    """
    This function split the data.
    Warning: Make sure that you have images in data_all folder
    """
    data_all_path = setting.data_all_path
    val_path = setting.val_path
    train_path = setting.train_path
    test_path = setting.test_path
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    counter = 0
    for (_, _, filenames) in os.walk(data_all_path):
        for _ in filenames:
            counter = counter + 1
    
    test = int(counter * setting.test_split)
    validation = int(counter * setting.val_split)
    train = int(counter * setting.train_split)

    for (_, _, filenames) in os.walk(data_all_path):
        for i, filename in enumerate(filenames):
            src = f"{data_all_path}/{filename}"
            if i in range(train):
                dst = f"{train_path}/{filename}"
            elif i in range(train, train + test):
                dst = f"{test_path}/{filename}"
            else:
                dst = f"{val_path}/{filename}"

            try:
                shutil.copyfile(src, dst)
            except shutil.Error as err:
                print(err.errno)


# def normalize_data(path_src: Path, path_dst: Path):
#     """
#     This function should normalize images
#     and create folder for normalized images.
#     """
#     os.makedirs(path_dst, exist_ok=True)
#     for (_, _, filenames) in os.walk(path_src):
#         scaler = MinMaxScaler()
#         for filename in filenames:
#             try:
#                 img = cv2.imread(f"{path_src}/{filename}",0)
#                 scaler.fit(img)
#                 norm_img = scaler.transform(img)
#                 cv2.imwrite(f"{path_dst}/{filename}", norm_img)
#             except cv2.error as err:
#                 print(err.msg)




def prepare_data(setting):
    split_date(setting)
    # normalize_data(setting.train_path, setting.normalize_train_path)
    # normalize_data(setting.test_path, setting.normalize_test_path)
    # normalize_data(setting.val_path, setting.normalize_val_path)
