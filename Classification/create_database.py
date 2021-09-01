import os
import cv2 as cv
import random

TRAIN_FOLDER = 'Labeled/train'
VALID_FOLDER = 'Labeled/valid'
IMAGES_SUBFOLDER = 'images'
TXT_SUBFOLDER = 'txts'
CROPPED_SUBFOLDER = 'cropped'
TRANSFORMED_SUBFOLDER = 'transformed'
CLASS_TXT_SUBFOLDER = 'class_txts'
IMAGES_EXT = 'JPG'
ANNOTS_EXT = 'xml'
TXT_EXT = 'txt'
OUT_SIZE = (64, 64)
BCKGRND_CLASS = 5
NUM_EMPTY_TRAIN = 200
NUM_EMPTY_VALID = 40


def get_filenames(folder):
    filenames = os.listdir(folder)
    for i in range(len(filenames)):
        filenames[i] = os.path.splitext(filenames[i])[0]
    random.shuffle(filenames)
    return filenames


ws = []
hs = []


def transform(cropped):
    return cv.resize(cropped, OUT_SIZE)


def create_database():
    for working_folder in [TRAIN_FOLDER, VALID_FOLDER]:
        empty_to_create = NUM_EMPTY_TRAIN if working_folder == TRAIN_FOLDER else NUM_EMPTY_VALID
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
            if len(insects) == 0 and empty_to_create > 0:
                cropped_name = f'{working_folder}/{CROPPED_SUBFOLDER}/{name}_empt.{IMAGES_EXT}'
                transformed_name = f'{working_folder}/{TRANSFORMED_SUBFOLDER}/{name}_empt.{IMAGES_EXT}'
                class_txt_name = f'{working_folder}/{CLASS_TXT_SUBFOLDER}/{name}_empt.{TXT_EXT}'
                print(f'\t{cropped_name}')
                insect_class = BCKGRND_CLASS  # Background class
                xmin = random.randint(1, w - OUT_SIZE[0] - 1)
                xmax = xmin + OUT_SIZE[0]
                ymin = random.randint(1, h - OUT_SIZE[1] - 1)
                ymax = ymin + OUT_SIZE[1]
                cropped = img[ymin:ymax + 1, xmin:xmax + 1, :]
                cv.imwrite(cropped_name, cropped)
                transformed = transform(cropped)
                cv.imwrite(transformed_name, transformed)
                with open(class_txt_name, 'w') as f:
                    f.write(f'{insect_class}\n')
                empty_to_create -= 1
            for i, insect in enumerate(insects):
                cropped_name = f'{working_folder}/{CROPPED_SUBFOLDER}/{name}_{i}.{IMAGES_EXT}'
                transformed_name = f'{working_folder}/{TRANSFORMED_SUBFOLDER}/{name}_{i}.{IMAGES_EXT}'
                class_txt_name = f'{working_folder}/{CLASS_TXT_SUBFOLDER}/{name}_{i}.{TXT_EXT}'
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
                with open(class_txt_name, 'w') as f:
                    f.write(f'{insect_class}\n')
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
        TRANSFORMED_SUBFOLDER,
        CLASS_TXT_SUBFOLDER
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
