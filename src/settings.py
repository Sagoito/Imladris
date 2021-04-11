from pathlib import Path

"""
This is only Setting class
"""


class Setting:
    def __init__(self, path: Path):
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
        self.user_image = "static/user_image"
        # split parameters
        self.train_split = 0.5
        self.test_split = 0.3
        self.val_split = 0.2