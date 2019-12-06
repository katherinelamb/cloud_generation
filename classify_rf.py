import os
import numpy as np
import pandas as pd

def main():
    BASE_DIR = os.path.dirname((os.path.dirname(__file__)))
    CROPS_DIR = os.path.join(BASE_DIR, 'inputs/crops/train_crops')
    TRAIN_CSV = os.path.join(BASE_DIR, 'inputs/crops/train_crops.csv')

if __name__ == '__main__':
    main()
