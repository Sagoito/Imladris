from pathlib import Path

"""
This is only Setting class
"""


class Setting:
    def __init__(self, path: Path,
                 train_split=0.5,
                 test_split=0.3,
                 val_split=0.2):
        # all images
        self.data_all_path = Path(f"{path}/images_all")
        # images after split
        self.val_path = Path(f"{path}/val2017")
        self.train_path = Path(f"{path}/train2017")
        self.test_path = Path(f"{path}/test2017")
        # normalize images
        self.normalize_train_path = Path(f"{path}/norm_train2017")
        self.normalize_test_path = Path(f"{path}/norm_test2017")
        self.normalize_val_path = Path(f"{path}/norm_val2017")
        # temporary user files
        self.user_image = "static/user_image/"
        # split parameters
        self.train_split = train_split
        self.test_split = test_split
        self.val_split = val_split
