import numpy as np
from glob import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from imgaug import SegmentationMapsOnImage
import tqdm
import csv

# helper function to get a string of labels for the picture
def get_labels(image_id):
    ''' Function to get the labels for the image by name'''
    im_df = train_df[train_df['Image'] == image_id].fillna('-1')
    im_df = im_df[im_df['EncodedPixels'] != '-1'].groupby('Label').count()

    index = im_df.index
    all_labels = ['Fish', 'Flower', 'Gravel', 'Sugar']

    labels = ''

    for label in all_labels:
        if label in index:
            labels = labels + ' ' + label

    return labels

# function to plot a grid of images and their labels
def plot_training_images(width = 5, height = 2):
    """
    Function to plot grid with several examples of cloud images from train set.
    INPUT:
        width - number of images per row
        height - number of rows

    OUTPUT: None
    """

    # get a list of images from training set
    images = sorted(glob(TRAIN_PATH + '*.jpg'))

    fig, axs = plt.subplots(height, width, figsize=(width * 3, height * 3))

    # create a list of random indices
    rnd_indices = rnd_indices = [np.random.choice(range(0, len(images))) for i in range(height * width)]

    for im in range(0, height * width):
        # open image with a random index
        image = Image.open(images[rnd_indices[im]])

        i = im // width
        j = im % width

        # plot the image
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')
        axs[i,j].set_title(get_labels(images[rnd_indices[im]].split('/')[-1]))

    # set suptitle
    plt.suptitle('Sample images from the train set')
    plt.show()

def rle_to_mask(rle_string, width, height):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters:
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask

    Returns:
    numpy.array: numpy array of the mask
    '''

    rows, cols = height, width

    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img

def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False

def get_mask(line_id, shape = (2100, 1400)):
    '''
    Function to visualize the image and the mask.
    INPUT:
        line_id - id of the line to visualize the masks
        shape - image shape
    RETURNS:
        np_mask - numpy segmentation map
    '''
    # replace null values with '-1'
    im_df = train_df.fillna('-1')

    # convert rle to mask
    rle = im_df.loc[line_id]['EncodedPixels']
    if rle != '-1':
        np_mask = rle_to_mask(rle, shape[0], shape[1])
        np_mask = np.clip(np_mask, 0, 1)
    else:
        # empty mask
        np_mask = np.zeros((shape[0],shape[1]), dtype=np.uint8)

    return np_mask

# helper function to get segmentation mask for an image by filename
def get_mask_by_image_id(image_id, label):
    '''
    Function to visualize several segmentation maps.
    INPUT:
        image_id - filename of the image
    RETURNS:
        np_mask - numpy segmentation map
    '''
    im_df = train_df[train_df['Image'] == image_id.split('/')[-1]].fillna('-1')

    image = np.asarray(Image.open(image_id))

    rle = im_df[im_df['Label'] == label]['EncodedPixels'].values[0]
    if rle != '-1':
        np_mask = rle_to_mask(rle, np.asarray(image).shape[1], np.asarray(image).shape[0])
        np_mask = np.clip(np_mask, 0, 1)
    else:
        # empty mask
        np_mask = np.zeros((np.asarray(image).shape[0], np.asarray(image).shape[1]), dtype=np.uint8)

    return np_mask

def visualize_image_with_mask(line_id):
    '''
    Function to visualize the image and the mask.
    INPUT:
        line_id - id of the line to visualize the masks
    '''
    # replace null values with '-1'
    im_df = train_df.fillna('-1')

    # get segmentation mask
    np_mask = get_mask(line_id)
    mask = np.where(np_mask.T == 1)
    bbox = np.min(mask[0]), np.min(mask[1]), np.max(mask[0]), np.max(mask[1])

    # open the image
    image = Image.open(TRAIN_PATH + im_df.loc[line_id]['Image'])
    crop_image = image.crop(bbox)

    # create segmentation map
    segmap = SegmentationMapsOnImage(np_mask, np_mask.shape)

    # visualize the image and map
    side_by_side = np.hstack([
        segmap.draw_on_image(np.asarray(image))
    ]).reshape(np.asarray(image).shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    plt.title(im_df.loc[line_id]['Label'])

    ax.imshow(side_by_side)

def long_slice(line_id, img, label, outdir, csv_path, sliceHeight, sliceWidth):
    imageWidth, imageHeight = img.size # Get image dimensions
    num_divs_width = imageWidth // sliceWidth
    num_divs_height = imageHeight // sliceHeight
    left = (imageWidth - num_divs_width * sliceWidth) // 2 # Set the left-most edge
    upper = (imageHeight - num_divs_height * sliceHeight) // 2 # Set the top-most edge
    index = 0
    while (left < imageWidth):
        while (upper < imageHeight):
            # If the bottom and right of the cropping box overruns the image.
            if (upper + sliceHeight > imageHeight or left + sliceWidth > imageWidth):
                break
            bbox = (left, upper, left + sliceWidth, upper + sliceHeight)
            working_slice = img.crop(bbox) # Crop image based on created bounds
            np_img = np.array(working_slice)
            if (np.sum([np_img == 0]) / np_img.size) > 0.005: break
            # Save your new cropped image.
            filename = '{}_{}.jpg'.format(line_id, index)
            out_path = os.path.join(outdir, filename)
            working_slice.thumbnail((128, 128), Image.ANTIALIAS)
            working_slice.save(out_path)
            with open(csv_path, 'a') as file:
                file.write('{},{}\n'.format(filename, label))
            index += 1
            upper += sliceHeight # Increment the horizontal position
        left += sliceWidth # Increment the vertical position
        upper = (imageHeight - num_divs_height * sliceHeight) // 2

def crop_image_with_mask(line_id, im_df, img_path, outdir, csv_path):
    # get segmentation mask
    np_mask = get_mask(line_id).T
    mask = np.where(np_mask == 1)
    if len(mask[0]) == 0 or len(mask[1]) == 0: return None
    bbox = np.min(mask[0]), np.min(mask[1]), np.max(mask[0]), np.max(mask[1])

    # open the image
    image = Image.open(img_path + im_df.loc[line_id]['Image']).crop(bbox)

    width, height = image.size
    new_width = new_height = 512

    if (width < new_width) or (height < new_height): return None
    long_slice(line_id, image, im_df.loc[line_id]['Label'], outdir, csv_path, new_height, new_width)

BASE_DIR = os.path.dirname((os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'inputs/understanding_cloud_organization')
CROP_DIR = os.path.join(BASE_DIR, 'inputs/crops')

# set paths to train and test image datasets
TRAIN_PATH = os.path.join(DATA_DIR, 'train_images/')
TEST_PATH = os.path.join(DATA_DIR, 'test_images/')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train.csv')

TRAIN_CROP_PATH = os.path.join(CROP_DIR, 'train_crops/')
TEST_CROP_PATH = os.path.join(CROP_DIR, 'test_crops/')
TRAIN_CSV_CROP_PATH = os.path.join(CROP_DIR, 'train_crops.csv')
TEST_CSV_CROP_PATH = os.path.join(CROP_DIR, 'test_crops.csv')
os.makedirs(TRAIN_CROP_PATH, exist_ok=True)
os.makedirs(TEST_CROP_PATH, exist_ok=True)

# load dataframe with train labels
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_fns = sorted(glob(TRAIN_PATH + '*.jpg'))

def main():
    # split column
    split_df = test_df["Image_Label"].str.split("_", n = 1, expand = True)
    # add new columns to train_df
    test_df['Image'] = split_df[0]
    test_df['Label'] = split_df[1]
    im_df = test_df.fillna(-1)

    print("Generating Test Crops")
    with open(TRAIN_CSV_CROP_PATH, 'w') as file:
        file.write('Image,Label\n')
    for line_id, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
        crop_image_with_mask(line_id, im_df, TEST_PATH, TEST_CROP_PATH, TEST_CSV_CROP_PATH)
        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.axis('off')
        # plt.title(row['Label'])
        # ax.imshow(image)
        # plt.show()

if __name__ == '__main__':
    main()
