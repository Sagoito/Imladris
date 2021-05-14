from src.data_preprocesing import prepare_data
from src.settings import Setting
import os
from pathlib import Path


if __name__ == "__main__":
    os.makedirs(Path('src/images_all'), exist_ok=True)
    setting = Setting('src')
    prepare_data(setting)
