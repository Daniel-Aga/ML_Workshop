import os
import random
import shutil
from PIL import Image
import re

TRAIN_FRAC = 0.75
VALID_FRAC = 0.15
# TEST_FRAC = 1 - TRAIN_FRAC - VALID_FRAC

DATA_FOLDER = './yolo'
TRAIN_FOLDER = './train'
VALID_FOLDER = './valid'
TEST_FOLDER = './test'

IMAGES_SUBFOLDER = 'images'
ANNOTS_SUBFOLDER = 'annots'
TXT_SUBFOLDER = 'txts'

IMAGES_EXT = 'JPG'
ANNOTS_EXT = 'xml'
TXT_EXT = 'txt'

MAX_SIZE = (720, 720)

def clean():
    confirm = input('Clean (Y/N)? ')
    if confirm.lower() != 'y':
        print('Canceled!')
        return False
    for fld in [f'{TRAIN_FOLDER}/{IMAGES_SUBFOLDER}',
                f'{TRAIN_FOLDER}/{ANNOTS_SUBFOLDER}',
                f'{TRAIN_FOLDER}/{TXT_SUBFOLDER}',
                f'{VALID_FOLDER}/{IMAGES_SUBFOLDER}',
                f'{VALID_FOLDER}/{ANNOTS_SUBFOLDER}',
                f'{VALID_FOLDER}/{TXT_SUBFOLDER}',
                f'{TEST_FOLDER}/{IMAGES_SUBFOLDER}',
                f'{TEST_FOLDER}/{ANNOTS_SUBFOLDER}',
                f'{TEST_FOLDER}/{TXT_SUBFOLDER}'
                ]:
        for filename in os.listdir(fld):
            file_path = os.path.join(fld, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    print('Cleaned!')
    return True

def get_filenames(check=True):
    filenames = os.listdir(DATA_FOLDER)
    names = set()
    for filename in filenames:
        names.add(os.path.splitext(filename)[0])
    if check:
        for name in names:
            if not os.path.isfile(f'{DATA_FOLDER}/{name}.{IMAGES_EXT}') or not os.path.isfile(
                    f'{DATA_FOLDER}/{name}.{ANNOTS_EXT}') or not os.path.isfile(f'{DATA_FOLDER}/{name}.{TXT_EXT}'):
                raise FileNotFoundError(f'Problem with {name}')
        print('All files fine!')
    return list(names)

def transform(img, annot, txt):
    w, h = img.size
    ratio = min(MAX_SIZE[0]/w, MAX_SIZE[1]/h, 1)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    new_img = img.resize((new_w, new_h))
    annot = re.sub(r'<xmin>(\d+)', lambda x: '<xmin>'+str(int(int(x.group(1)) * ratio)), annot)
    annot = re.sub(r'<xmax>(\d+)', lambda x: '<xmax>'+str(int(int(x.group(1)) * ratio)), annot)
    annot = re.sub(r'<ymin>(\d+)', lambda x: '<ymin>'+str(int(int(x.group(1)) * ratio)), annot)
    annot = re.sub(r'<ymax>(\d+)', lambda x: '<ymax>'+str(int(int(x.group(1)) * ratio)), annot)
    annot = re.sub(r'<width>(\d+)', lambda x: '<width>'+str(new_w), annot)
    annot = re.sub(r'<height>(\d+)', lambda x: '<height>'+str(new_h), annot)
    return new_img, annot, txt

def write_folder(names, dest_folder):
    for name in names:
        img = Image.open(f'{DATA_FOLDER}/{name}.{IMAGES_EXT}')
        with open(f'{DATA_FOLDER}/{name}.{ANNOTS_EXT}', 'r') as f:
            annot = f.read()
        with open(f'{DATA_FOLDER}/{name}.{TXT_EXT}',  'r') as f:
            txt = f.read()
        new_img, new_annot, new_txt = transform(img, annot, txt)
        new_img.save(f'{dest_folder}/{IMAGES_SUBFOLDER}/{name}.{IMAGES_EXT}')
        with open(f'{dest_folder}/{ANNOTS_SUBFOLDER}/{name}.{ANNOTS_EXT}','w') as f:
            f.write(new_annot)
        with open(f'{dest_folder}/{TXT_SUBFOLDER}/{name}.{TXT_EXT}','w') as f:
            f.write(new_txt)
        img.close()
        new_img.close()

def split():
    if not clean():
        return
    names = get_filenames()
    num_samples = len(names)
    num_train = int(num_samples * TRAIN_FRAC)
    num_valid = int(num_samples * VALID_FRAC)
    num_test = num_samples - num_train - num_valid
    print(f'Train: {num_train}, Valid: {num_valid}, Test: {num_test}. Total: {num_samples}')
    train_names = random.sample(names, num_train)
    names_left = [n for n in names if n not in train_names]
    valid_names = random.sample(names_left, num_valid)
    test_names = [n for n in names_left if n not in valid_names]
    assert len(test_names) == num_test
    write_folder(train_names, TRAIN_FOLDER)
    write_folder(valid_names, VALID_FOLDER)
    write_folder(test_names, TEST_FOLDER)
    # for name in train_names:
    #     shutil.copy(f'{DATA_FOLDER}/{name}.{IMAGES_EXT}', f'{TRAIN_FOLDER}/{IMAGES_SUBFOLDER}')
    #     shutil.copy(f'{DATA_FOLDER}/{name}.{ANNOTS_EXT}', f'{TRAIN_FOLDER}/{ANNOTS_SUBFOLDER}')
    # for name in valid_names:
    #     shutil.copy(f'{DATA_FOLDER}/{name}.{IMAGES_EXT}', f'{VALID_FOLDER}/{IMAGES_SUBFOLDER}')
    #     shutil.copy(f'{DATA_FOLDER}/{name}.{ANNOTS_EXT}', f'{VALID_FOLDER}/{ANNOTS_SUBFOLDER}')
    # for name in test_names:
    #     shutil.copy(f'{DATA_FOLDER}/{name}.{IMAGES_EXT}', f'{TEST_FOLDER}/{IMAGES_SUBFOLDER}')
    #     shutil.copy(f'{DATA_FOLDER}/{name}.{ANNOTS_EXT}', f'{TEST_FOLDER}/{ANNOTS_SUBFOLDER}')
    print('DONE')

def get_min_max():
    names = get_filenames()
    mn = 20000
    mx = -1
    for name in names:
        img = Image.open(f'{DATA_FOLDER}/{name}.{IMAGES_EXT}')
        w, h = img.size
        ratio = min(MAX_SIZE[0] / w, MAX_SIZE[1] / h, 1)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        mn = min(mn, new_w, new_h)
        mx = max(mx, new_w, new_h)
        img.close()
    print(mn, mx)

if __name__ == '__main__':
    split()
    get_min_max()