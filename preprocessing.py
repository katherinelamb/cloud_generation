import numpy as np
import glob
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname((os.path.dirname(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, 'inputs/understanding_cloud_organization')

# set paths to train and test image datasets
TRAIN_PATH = os.path.join(DATA_DIR, 'train_images/')
TEST_PATH = os.path.join(DATA_DIR, 'test_images/')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train.csv')

# load dataframe with train labels
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_fns = sorted(glob(TRAIN_PATH + '*.jpg'))

print(train_df.head())
