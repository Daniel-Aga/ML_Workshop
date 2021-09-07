import numpy as np
import os
from PIL import Image, ImageDraw
import cv2 as cv

UNLABELED_FOLDER = 'unlabeled_hanadiv'
OUTPUT_FOLDER = 'outs_hanadiv'
GRAYCSCALE_OUT = 'outs_hanadiv_gray'
THRESHOLD_OUT = 'outs_hanadiv_thresh'
CROPPED_OUT = 'outs_hanadiv_cropped'
IMAGES_EXT = 'JPG'
TRIM_BOTTOM = 230
TRIM_LEFT = 0
TRIM_RIGHT = 0
TRIM_TOP = 0
SIZE_FACTOR = 0.25
NUM_IMAGES_TO_PROCESS = 1000
NUM_IMAGES_FOR_SVD = 100
SMALL_SPACE_DIM = 10
X_NP_FILE = 'tmp/X.npy'
S_NP_FILE = 'tmp/S.npy'
Vt_NP_FILE = 'tmp/Vt.npy'
U_NP_FILE = 'tmp/U.npy'
P_NP_FILE = 'tmp/P.npy'
Q_NP_FILE = 'tmp/Q.npy'
APPLY_TO_RESULT = lambda x: (x + 200) * 255 / 300
# APPLY_TO_RESULT = lambda x: (x + 150) * 255 / 200
SQUARE_SIZE = (16, 16)
LPF_FILE = 'lpf.csv'
DO_LOW_PASS = False
THRESHOLD = 130
NUM_PIXELS_SMALLER_THAN_THRESH = 3
NUM_PIXELS_TO_CHECK = 10

lpf = None

def init():
    global lpf
    lpf = np.genfromtxt(LPF_FILE, delimiter=',')
    pass

def PIL_to_cv2(img):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

def cv2_to_PIL(img):
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

def shrink(img):
    w, h = img.size
    new_w = int(w * SIZE_FACTOR)
    new_h = int(h * SIZE_FACTOR)
    new_img = img.resize((new_w, new_h))
    return new_img

def crop(img):
    w, h = img.size
    return img.crop((TRIM_LEFT, TRIM_TOP, w - TRIM_RIGHT, h - TRIM_BOTTOM))

def low_pass(img):
    if not DO_LOW_PASS:
        return img
    # old = img.copy()
    # img.save('test.jpg')
    img = PIL_to_cv2(img)
    img = cv.filter2D(img, -1, lpf)

    # old = PIL_to_cv2(old)
    # diff = img - old
    # diff = Image.fromarray(diff)
    # diff.save('diff.jpg')

    img = cv2_to_PIL(img)
    # img.save('test2.jpg')

    return img

def preprocess(img):
    img = crop(img)
    img = low_pass(img)
    img = shrink(img)
    return img

def loadX(files, save=True, force=False):
    if not force and os.path.isfile(X_NP_FILE):
        return np.load(X_NP_FILE)
    X_lst = []
    c = 0
    # for fname in os.listdir(fld):
    for fname in files:
        print(f'{c + 1}. {fname}')
        img = Image.open(f'{UNLABELED_FOLDER}/{fname}')
        # img.show()
        # img = img.filter(ImageFilter.GaussianBlur(20))
        # img = img.filter(ImageFilter.MinFilter(5))
        # img.show()
        new_img = preprocess(img)
        arr = np.asarray(new_img).reshape(1, -1)
        X_lst.append(arr)
        img.close()
        new_img.close()
        c += 1
    X = np.vstack(X_lst).astype('float64')
    # X -= X.mean(axis=0, keepdims=True)
    if save:
        np.save(X_NP_FILE, X)
    return X


def get_svd_vecs(X, k, save=True, force=False):
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
    return P @ (Q @ v)


def get_orthogonal_component(P, Q, v):
    return v - get_projection(P, Q, v)


def mat_argmin(arr):
    return np.unravel_index(arr.argmin(), arr.shape)


def is_insect_in_square(arr, arr_gray, sqr):
    sub_image = arr_gray[sqr[0]:sqr[0] + SQUARE_SIZE[0], sqr[1]:sqr[1] + SQUARE_SIZE[1]]
    return (sub_image < THRESHOLD).sum() >= NUM_PIXELS_SMALLER_THAN_THRESH

def in_sqr(sqr, pt):
    """ Square is of size SQUARE_SIZE. """
    return sqr[0] <= pt[0] <= sqr[0] + SQUARE_SIZE[0] and sqr[1] <= pt[1] <= sqr[1] + SQUARE_SIZE[1]

def intersect_squares(sqr1, sqr2):
    """ Squares are of size SQUARE_SIZE. """
    return in_sqr(sqr2, (sqr1[0], sqr1[1])) or in_sqr(sqr2, (sqr1[0] + SQUARE_SIZE[0], sqr1[1])) or in_sqr(sqr2, (sqr1[0], sqr1[1] + SQUARE_SIZE[1])) or in_sqr(sqr2, (sqr1[0] + SQUARE_SIZE[0], sqr1[1] + SQUARE_SIZE[1]))

def insect_square(arr, arr_gray):
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


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


mins = []
maxes = []


def output(P, Q, files):

    c = 0
    for fname in files:
        print(f'{c + 1}. {fname}')
        img = Image.open(f'{UNLABELED_FOLDER}/{fname}')
        new_img = preprocess(img)
        new_w, new_h = new_img.size
        arr = np.asarray(new_img).reshape(-1)
        res_np = get_orthogonal_component(P, Q, arr).reshape((new_h, new_w, 3))
        # print(f'\tMIN: {res_np.min()}, MAX: {res_np.max()}'); mins.append(res_np.min()); maxes.append(res_np.max());
        # res_np -= res_np.min()
        # res_np = (res_np / res_np.max()) * 255
        res_np = APPLY_TO_RESULT(res_np)
        res_np = res_np.astype('uint8')
        gray_np = rgb2gray(res_np)

        res_img = Image.fromarray(res_np)
        res_img.save(f'{OUTPUT_FOLDER}/{fname}')
        gray_image = res_img.convert('L')
        # gray_image = Image.fromarray(gray_np)
        gray_image.save(f'{GRAYCSCALE_OUT}/{fname}')

        res_img.close()

        gray_image.close()
        img.close()
        new_img.close()
        c += 1
        # if 0 < num_images <= c:
        #     break


def analyze_insects(files):
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

        sqr_img.save(f'{THRESHOLD_OUT}/{fname}')
        sqr_img.close()
        img.close()
        new_img.close()
        res_img.close()
        gray_img.close()

    return total_insects

def clean(force=False, folders=None):
    if folders is None:
        folders=[OUTPUT_FOLDER, GRAYCSCALE_OUT, THRESHOLD_OUT, CROPPED_OUT]
    if not force:
        confirm = input('Clean (Y/N)? ')
        if confirm.lower() != 'y':
            print('Canceled!')
            return False
    for fld in folders:
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
        files = all_files[i:i+NUM_IMAGES_FOR_SVD]
        print(f'Loading images...')
        # X = loadX(FOLDER, NUM_IMAGES_TO_PROCESS, save=False)
        X = loadX(files, save=False)
        print(f'Processing SVD...')
        # rows_to_take = np.random.choice(NUM_IMAGES_TO_PROCESS, NUM_IMAGES_FOR_SVD, replace=False)
        # X = X[rows_to_take, ...]
        S, V = get_svd_vecs(X, SMALL_SPACE_DIM, save=False)
        P, Q = get_proj_mat(V, save=False)
        # clean(True)
        print(f'Processing images...')
        # output(P, Q, FOLDER, NUM_IMAGES_TO_PROCESS)
        output(P, Q, files)
        total_insects += analyze_insects(files)
    print(f'Done!')
    print(f'Found a total of {total_insects} insects.')

def only_analyze():
    init()
    clean(True, [THRESHOLD_OUT, CROPPED_OUT])
    all_files = os.listdir(UNLABELED_FOLDER)
    total_insects = 0
    for i in range(0, min(len(all_files), NUM_IMAGES_TO_PROCESS), NUM_IMAGES_FOR_SVD):
        files = all_files[i:i+NUM_IMAGES_FOR_SVD]
        total_insects += analyze_insects(files)
    print(f'Done!')
    print(f'Found a total of {total_insects} insects.')


if __name__ == '__main__':
    only_analyze()
    # main()