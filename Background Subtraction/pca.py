import numpy as np
import os
from PIL import Image, ImageDraw
import cv2 as cv
from pathlib import Path

### General:
ONLY_ANALYZE = True  # Don't remove background; Use previously created foreground images and only analyze bounding boxes.

### INPUT:
UNLABELED_FOLDER = 'unlabeled_hanadiv'  # Input folder: unlabeled images.
# UNLABELED_FOLDER = 'unlabeled_mishmar'
NUM_IMAGES_TO_PROCESS = 1000  # Total number of images to process.

### OUTPUT:
OUTPUT_FOLDER = 'outs_hanadiv'  # Output folder: foreground images.
GRAYCSCALE_OUT = 'outs_hanadiv_gray'  # Output folder: foreground images in grayscale.
BOXES_OUT = 'outs_hanadiv_thresh'  # Output folder: images with detected bounding boxes on them.
CROPPED_OUT = 'outs_hanadiv_cropped'  # Output folder: cropped bounding boxes.
# OUTPUT_FOLDER = 'outs_mishmar'
# GRAYCSCALE_OUT = 'outs_mishmar_gray'
# BOXES_OUT = 'outs_mishmar_thresh'
# CROPPED_OUT = 'outs_mishmar_cropped'

### Constants:
IMAGES_EXT = 'JPG'

### Preprocessing:
TRIM_BOTTOM = 230  # The height (in pixels) to crop from the bottom - the height of the watermark.
# TRIM_BOTTOM = 160
TRIM_LEFT = 0  # The width (in pixels) to crop from the left.
TRIM_RIGHT = 0  # The width (in pixels) to crop from the right.
TRIM_TOP = 0  # The height (in pixels) to crop from the top.
SIZE_FACTOR = 0.25  # The factor of resizing the images in the preprocessing stage.
DO_LOW_PASS = False  # Weather to run a low pass filter in the preprocessing stage.
LPF_FILE = 'lpf.csv'  # File containing the low pass filter. Ignored if DO_LOW_PASS = False.

### SVD:
NUM_IMAGES_FOR_SVD = 100  # The parameter 'N' in the SVD stage.
SMALL_SPACE_DIM = 10  # The parameter 'r' in the SVD stage.
APPLY_TO_RESULT = lambda x: (
                                    x + 150) * 255 / 200  # The linear function to apply to foreground pixel values in order to create an image.
# APPLY_TO_RESULT = lambda x: (x + 200) * 255 / 300


### Bounding Box Extraction:
SQUARE_SIZE = (16, 16)  # Wanted box size (in the smaller, resized image).
THRESHOLD = 115  # The parameter 't' in the box extraction stage.
# THRESHOLD = 110
NUM_PIXELS_SMALLER_THAN_THRESH = 3  # The parameter 'l' in the box extraction stage.
NUM_PIXELS_TO_CHECK = 10  # The parameter 'k' in the box extraction stage.
# NUM_PIXELS_TO_CHECK = 20


### Deprecated:
X_NP_FILE = 'tmp/X.npy'  # File to save intermediate results (deprecated).
S_NP_FILE = 'tmp/S.npy'  # File to save intermediate results (deprecated).
Vt_NP_FILE = 'tmp/Vt.npy'  # File to save intermediate results (deprecated).
U_NP_FILE = 'tmp/U.npy'  # File to save intermediate results (deprecated).
P_NP_FILE = 'tmp/P.npy'  # File to save intermediate results (deprecated).
Q_NP_FILE = 'tmp/Q.npy'  # File to save intermediate results (deprecated).

lpf = None


def init():
    """ Init the low pass filter. """
    if not DO_LOW_PASS:
        return
    global lpf
    lpf = np.genfromtxt(LPF_FILE, delimiter=',')


def PIL_to_cv2(img):
    """
    Converts a PIL image to OpenCV image. 
    :param img: The input image.
    :return: The converted image.
    """
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def cv2_to_PIL(img):
    """
    Converts an OpenCV image to a PIL image. 
    :param img: The input image.
    :return: The converted image.
    """
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))


def shrink(img):
    """
    Resizes the image according to SIZE_FACTOR. 
    :param img: The input image.
    :return: The resized image.
    """
    w, h = img.size
    new_w = int(w * SIZE_FACTOR)
    new_h = int(h * SIZE_FACTOR)
    new_img = img.resize((new_w, new_h))
    return new_img


def crop(img):
    """
    Crops the image according to TRIM_*.
    :param img: The input image.
    :return: The cropped image.
    """
    w, h = img.size
    return img.crop((TRIM_LEFT, TRIM_TOP, w - TRIM_RIGHT, h - TRIM_BOTTOM))


def low_pass(img):
    """
    Performs the low pass filter. 
    :param img: The input image.
    :return: The result image after the filter.
    """
    if not DO_LOW_PASS:
        return img
    img = PIL_to_cv2(img)
    img = cv.filter2D(img, -1, lpf)

    img = cv2_to_PIL(img)
    return img


def preprocess(img):
    """
    Performs the preprocessing stage.
    :param img: The input image.
    :return: The preprocessed image.
    """
    img = crop(img)
    img = low_pass(img)
    img = shrink(img)
    return img


def loadX(files, save=True, force=False):
    """
    Loads the matrix X containing the (preprocessed) images in its rows.
    :param files: Filenames to load: images name in UNLABELED_FOLDER.
    :param save: Weather to save intermediate results. Deprecated.
    :param force: Weather to force recalculating even if intermediate result exists. Deprecated.
    :return: The matrix X.
    """
    ### Deprecated features:
    save = False

    if not force and os.path.isfile(X_NP_FILE):
        return np.load(X_NP_FILE)
    X_lst = []
    c = 0
    for fname in files:
        print(f'{c + 1}. {fname}')
        img = Image.open(f'{UNLABELED_FOLDER}/{fname}')
        new_img = preprocess(img)
        arr = np.asarray(new_img).reshape(1, -1)
        X_lst.append(arr)
        img.close()
        new_img.close()
        c += 1
    X = np.vstack(X_lst).astype('float64')
    if save:
        np.save(X_NP_FILE, X)
    return X


def get_svd_vecs(X, k, save=True, force=False):
    """
    Performs SVD on the given matrix.
    :param X: The matrix X containing the preprocessed images in its rows.
    :param k: The number of singular vectors to take for the subspace.
    :param save: Weather to save intermediate results. Deprecated.
    :param force: Weather to force recalculating even if intermediate result exists. Deprecated.
    :return: The first k singular values and the first k singular vectors.
    """
    ### Deprecated features:
    save = False

    if not force and os.path.isfile(U_NP_FILE) and os.path.isfile(S_NP_FILE) and os.path.isfile(Vt_NP_FILE):
        U = np.load(U_NP_FILE)
        S = np.load(S_NP_FILE)
        Vt = np.load(Vt_NP_FILE)
        return S[:k], Vt.T[:, :k]

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if save:
        np.save(U_NP_FILE, U)
        np.save(S_NP_FILE, S)
        np.save(Vt_NP_FILE, Vt)
    return S[:k], Vt.T[:, :k]


def get_proj_mat(V, save=True, force=False):
    """
    Computes the orthogonal projection transformation to the subspace spanned by the given k vectors.
    :param V: A matrix holding the k vectors in its columns.
    :param save: Weather to save intermediate results. Deprecated.
    :param force: Weather to force recalculating even if intermediate result exists. Deprecated.
    :return: Matrices P and Q, such that "v -> P @ Q @ v" is the orthogonal projection.
    """
    if not force and os.path.isfile(P_NP_FILE):
        P = np.load(P_NP_FILE)
        Q = np.load(Q_NP_FILE)
        return P, Q
    P = V @ np.linalg.inv(V.T @ V)
    Q = V.T
    if save:
        np.save(P_NP_FILE, P)
        np.save(Q_NP_FILE, Q)
    return P, Q


def get_projection(P, Q, v):
    """
    Performs the projection defined by the matrices P and Q on the vector v.
    :param P: Defines the orthogonal projection.
    :param Q: Defines the orthogonal projection.
    :param v: The input vector.
    :return: The resulting projected vector.
    """
    return P @ (Q @ v)


def get_orthogonal_component(P, Q, v):
    """
    Returns the orthogonal component of the vector v, with respect to the subspace defined by the orthogonal projection P @ Q.
    :param P: Defines the orthogonal projection.
    :param Q: Defines the orthogonal projection.
    :param v: The input vector.
    :return: The orthogonal component.
    """
    return v - get_projection(P, Q, v)


def is_insect_in_square(arr, arr_gray, sqr):
    """
    Decides if there is an insect in the given square.
    :param arr: The foreground image.
    :param arr_gray: The foreground image in grayscale.
    :param sqr: The upper-left corner of the proposed square.
    :return: True if the model thinks there is an insect in this square; False otherwise.
    """
    sub_image = arr_gray[sqr[0]:sqr[0] + SQUARE_SIZE[0], sqr[1]:sqr[1] + SQUARE_SIZE[1]]
    return (sub_image < THRESHOLD).sum() >= NUM_PIXELS_SMALLER_THAN_THRESH


def in_sqr(sqr, pt):
    """
    Checks if the given point is inside the given square.
    :param sqr: The upper-left corner of the square.
    :param pt: The input point.
    :return: True if the given point is inside the given square; False otherwise.
    """
    return sqr[0] <= pt[0] <= sqr[0] + SQUARE_SIZE[0] and sqr[1] <= pt[1] <= sqr[1] + SQUARE_SIZE[1]


def intersect_squares(sqr1, sqr2):
    """
    Checks if the given squares intersect.
    :param sqr1: The upper-left corner of the first square.
    :param sqr2: The upper-left corner of the second square.
    :return: True if the given squares intersect; False otherwise.
    """
    return in_sqr(sqr2, (sqr1[0], sqr1[1])) or in_sqr(sqr2, (sqr1[0] + SQUARE_SIZE[0], sqr1[1])) or in_sqr(sqr2, (
        sqr1[0], sqr1[1] + SQUARE_SIZE[1])) or in_sqr(sqr2, (sqr1[0] + SQUARE_SIZE[0], sqr1[1] + SQUARE_SIZE[1]))


def insect_square(arr, arr_gray):
    """
    Returns a list of bounding boxes of insects in the image.
    :param arr: The foreground image.
    :param arr_gray: The foreground image in grayscale.
    :return: A list of computed squares, in the format [ymin, xmin, ymax, xmax].
    """
    pixels_to_check = np.unravel_index(np.argsort(arr_gray, axis=None)[:NUM_PIXELS_TO_CHECK], arr_gray.shape)
    pixels_to_check = list(zip(*pixels_to_check))
    potential_sqrs = [(pix[0] - SQUARE_SIZE[0] // 2, pix[1] - SQUARE_SIZE[1] // 2) for pix in pixels_to_check]
    insect_sqrs = [sqr for sqr in potential_sqrs if is_insect_in_square(arr, arr_gray, sqr)]
    final_sqrs = []
    for sqr in insect_sqrs:
        good = True
        for prev_sqr in final_sqrs:
            if intersect_squares(sqr, prev_sqr):
                good = False
                break
        if good:
            final_sqrs.append(sqr)
    return [[sqr[1], sqr[0], sqr[1] + SQUARE_SIZE[1], sqr[0] + SQUARE_SIZE[0]] for sqr in final_sqrs]


def output(P, Q, files):
    """
    Saves the foreground and grayscale foreground images for the given files, using the given orthogonal projection.
    :param P: Defines the orthogonal projection.
    :param Q: Defines the orthogonal projection.
    :param files: Filenames to load: images name in UNLABELED_FOLDER.
    :return: None
    """
    c = 0
    for fname in files:
        print(f'{c + 1}. {fname}')
        img = Image.open(f'{UNLABELED_FOLDER}/{fname}')
        new_img = preprocess(img)
        new_w, new_h = new_img.size
        arr = np.asarray(new_img).reshape(-1)
        res_np = get_orthogonal_component(P, Q, arr).reshape((new_h, new_w, 3))
        res_np = APPLY_TO_RESULT(res_np)
        res_np = res_np.astype('uint8')

        res_img = Image.fromarray(res_np)
        res_img.save(f'{OUTPUT_FOLDER}/{fname}')
        gray_image = res_img.convert('L')
        gray_image.save(f'{GRAYCSCALE_OUT}/{fname}')

        res_img.close()

        gray_image.close()
        img.close()
        new_img.close()
        c += 1


def analyze_insects(files):
    """
    Analyzes and saves the bounding boxes in the given images, given their foreground was already computed.
    :param files: Filenames to load: images name in UNLABELED_FOLDER.
    :return: The total number of insects found.
    """
    total_insects = 0
    for fname in files:
        print(f'{fname}')
        img = Image.open(f'{UNLABELED_FOLDER}/{fname}')
        new_img = preprocess(img)
        res_img = Image.open(f'{OUTPUT_FOLDER}/{fname}')
        gray_img = Image.open(f'{GRAYCSCALE_OUT}/{fname}')
        res_np = np.asarray(res_img).astype('uint8')
        gray_np = np.asarray(gray_img).astype('uint8')
        squares = insect_square(res_np, gray_np)
        sqr_img = new_img.copy()
        sqr_ind = 0
        for square in squares:
            print('\tFound insect!')
            d = ImageDraw.Draw(sqr_img)
            d.rectangle(square, outline='red', width=3)

            orig_sqr = map(lambda x: int(x / SIZE_FACTOR), square)
            cropped_img = crop(img).crop(orig_sqr)
            cropped_img.save(f'{CROPPED_OUT}/{os.path.splitext(fname)[0]}_{sqr_ind}.{IMAGES_EXT}')
            cropped_img.close()

            total_insects += 1
            sqr_ind += 1

        sqr_img.save(f'{BOXES_OUT}/{fname}')
        sqr_img.close()
        img.close()
        new_img.close()
        res_img.close()
        gray_img.close()

    return total_insects


def clean(force=False, folders=None):
    """
    Cleans the output directories. Also creates the directories if they don't exist.
    :param force: Weather to force deleting, without prompt.
    :param folders: A list of folders to clean. If None, all output folders will be cleaned.
    :return: True if successful; False otherwise.
    """
    if folders is None:
        folders = [OUTPUT_FOLDER, GRAYCSCALE_OUT, BOXES_OUT, CROPPED_OUT]
    if not force:
        confirm = input('Clean (Y/N)? ')
        if confirm.lower() != 'y':
            print('Canceled!')
            return False
    for fld in folders:
        Path(fld).mkdir(parents=True, exist_ok=True)
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
    init()
    clean(True)
    all_files = os.listdir(UNLABELED_FOLDER)
    total_insects = 0
    for i in range(0, min(len(all_files), NUM_IMAGES_TO_PROCESS), NUM_IMAGES_FOR_SVD):
        files = all_files[i:i + NUM_IMAGES_FOR_SVD]
        print(f'Loading images...')
        X = loadX(files, save=False)
        print(f'Processing SVD...')
        S, V = get_svd_vecs(X, SMALL_SPACE_DIM, save=False)
        P, Q = get_proj_mat(V, save=False)
        print(f'Processing images...')
        output(P, Q, files)
        total_insects += analyze_insects(files)
    print(f'Done!')
    print(f'Found a total of {total_insects} insects.')


def only_analyze():
    init()
    clean(True, [BOXES_OUT, CROPPED_OUT])
    all_files = os.listdir(UNLABELED_FOLDER)
    total_insects = 0
    for i in range(0, min(len(all_files), NUM_IMAGES_TO_PROCESS), NUM_IMAGES_FOR_SVD):
        files = all_files[i:i + NUM_IMAGES_FOR_SVD]
        total_insects += analyze_insects(files)
    print(f'Done!')
    print(f'Found a total of {total_insects} insects.')


if __name__ == '__main__':
    if ONLY_ANALYZE:
        only_analyze()
    else:
        main()
