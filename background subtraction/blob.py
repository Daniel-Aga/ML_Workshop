import cv2
import numpy as np

RATIO = 2 # how big the bounding boxes are
SIZE_FACTOR = 0.25 # same SIZE_FACTOR from pca
THRESHOLD = 135
UNLABELED_FOLDER = 'unlabeled_hanadiv'
OUTS_FOLDER = 'outs_hanadiv'
BOX_FOLDER = 'box_hanadiv'
# filename = 'WSCT0250.JPG'

def get_bounding_boxes(filename):
    outsname = f'{OUTS_FOLDER}/{filename}'
    realname = f'{UNLABELED_FOLDER}/{filename}'
    boxname = f'{BOX_FOLDER}/{filename}'
    im = cv2.imread(outsname, cv2.IMREAD_GRAYSCALE)
    width, height = im.shape[:2]

    ret, threshed = cv2.threshold(im, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('threshed', threshed)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2RGB)
    boxes = []

    for cnt in contours:
        min_x, min_y = np.amin(cnt, axis=0)[0]
        max_x, max_y = np.amax(cnt, axis=0)[0]
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        w = max(max_x - min_x, max_y - min_y)
        min_x, min_y = cx - w / 2, cy - w / 2
        max_x, max_y = cx + w / 2, cy + w / 2
        min_x = int(cx + RATIO * (min_x - cx))
        min_y = int(cy + RATIO * (min_y - cy))
        max_x = int(cx + RATIO * (max_x - cx))
        max_y = int(cy + RATIO * (max_y - cy))
        boxes.append((min_x, min_y, max_x, max_y))
    # return boxes

    
    # cv2.imshow('after boxes', threshed)
    # cv2.waitKey(0)
    real_im = cv2.imread(realname)
    for box in boxes:
        min_x, min_y, max_x, max_y = box
        # cv2.rectangle(threshed, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)
        cv2.rectangle(
            real_im,
            (min_x / SIZE_FACTOR, min_y / SIZE_FACTOR),
            (max_x / SIZE_FACTOR, max_y / SIZE_FACTOR),
            (0, 0, 255), 3
        )
    cv2.imwrite(boxname, real_im)

# get_bounding_boxes(filename)

for i in range(1, 461 + 1):
    get_bounding_boxes(f'WSCT{str(i).zfill(4)}.JPG')
