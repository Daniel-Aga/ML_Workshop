import numpy as np
import os
from PIL import Image, ImageDraw
import cv2 as cv
import torch

TRAIN_FOLDER = 'Labeled/train'
VALID_FOLDER = 'Labeled/valid'
IMAGES_SUBFOLDER = 'images'
TXT_SUBFOLDER = 'txts'
CROPPED_SUBFOLDER = 'cropped'
TRANSFORMED_SUBFOLDER = 'transformed'
IMAGES_EXT = 'JPG'
ANNOTS_EXT = 'xml'
TXT_EXT = 'txt'
OUT_SIZE = (64, 64)

def get_filenames(folder):
    filenames = os.listdir(folder)
    for i in range(len(filenames)):
        filenames[i] = os.path.splitext(filenames[i])[0]
    return filenames

ws = []
hs = []

def transform(cropped):
    return cv.resize(cropped, OUT_SIZE)

def create_database():
    for working_folder in [TRAIN_FOLDER, VALID_FOLDER]:
        img_folder = f'{working_folder}/{IMAGES_SUBFOLDER}'
        names = get_filenames(img_folder)
        for name in names:
            img_name = f'{working_folder}/{IMAGES_SUBFOLDER}/{name}.{IMAGES_EXT}'
            print(img_name)
            txt_name = f'{working_folder}/{TXT_SUBFOLDER}/{name}.{TXT_EXT}'
            img = cv.imread(img_name)
            h, w, three = img.shape
            with open(txt_name, 'r') as f:
                insects = f.readlines()
            insects = [x.strip() for x in insects]
            for i, insect in enumerate(insects):
                cropped_name = f'{working_folder}/{CROPPED_SUBFOLDER}/{name}_{i}.{IMAGES_EXT}'
                transformed_name = f'{working_folder}/{TRANSFORMED_SUBFOLDER}/{name}_{i}.{IMAGES_EXT}'
                print(f'\t{cropped_name}')
                insect_class, xmid, ymid, wins, hins = map(float, insect.split())
                xmin = (xmid - wins / 2) * w
                xmax = (xmid + wins / 2) * w
                ymin = (ymid - hins / 2) * h
                ymax = (ymid + hins / 2) * h
                insect_class, xmin, xmax, ymin, ymax = map(int, (insect_class, xmin, xmax, ymin, ymax))
                cropped = img[ymin:ymax + 1, xmin:xmax + 1, :]
                cv.imwrite(cropped_name, cropped)
                transformed = transform(cropped)
                cv.imwrite(transformed_name, transformed)
                ws.append(int(wins * w))
                hs.append(int(hins * h))

def clean(force=False):
    if not force:
        confirm = input('Clean (Y/N)? ')
        if confirm.lower() != 'y':
            print('Canceled!')
            return False
    to_clean = [f'{x}/{y}' for x in [TRAIN_FOLDER, VALID_FOLDER] for y in [
        CROPPED_SUBFOLDER,
        TRANSFORMED_SUBFOLDER
    ]]
    for fld in to_clean:
        for filename in os.listdir(fld):
            file_path = os.path.join(fld, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    print('Cleaned!')
    return True


def main():
    clean(True)
    print('Creating cropped images...')
    create_database()
    pass

if __name__ == '__main__':
    main()