import os
import numpy as np
import pandas as pd

def main():
    BASE_DIR = os.path.dirname((os.path.dirname(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'inputs/understanding_cloud_organization')
    CROP_DIR = os.path.join(BASE_DIR, 'inputs/crops')

    TRAIN_PATH = os.path.join(DATA_DIR, 'train_images/')
    TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train.csv')

if __name__ == '__main__':
    main()
