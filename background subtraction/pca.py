import numpy as np
import os
from PIL import Image, ImageDraw

FOLDER = 'unlabeled_mishmar'
OUTPUT_FOLDER = 'outs_mishmar'
GRAYCSCALE_OUT = 'outs_mishmar_gray'
THRESHOLD_OUT = 'outs_mishmar_thresh'
IMAGES_EXT = 'JPG'
TRIM_BOTTOM = 160
TRIM_LEFT = 0
TRIM_RIGHT = 0
TRIM_TOP = 0
SIZE_FACTOR = 0.25
NUM_IMAGES_TO_PROCESS = 500
NUM_IMAGES_FOR_SVD = 100
SMALL_SPACE_DIM = 10
X_NP_FILE = 'tmp/X.npy'
S_NP_FILE = 'tmp/S.npy'
Vt_NP_FILE = 'tmp/Vt.npy'
U_NP_FILE = 'tmp/U.npy'
P_NP_FILE = 'tmp/P.npy'
Q_NP_FILE = 'tmp/Q.npy'
APPLY_TO_RESULT = lambda x: (x + 200) * 255 / 300
SQUARE_SIZE = (16, 16)


def loadX(files, save=True, force=False):
    if not force and os.path.isfile(X_NP_FILE):
        return np.load(X_NP_FILE)
    X_lst = []
    c = 0
    # for fname in os.listdir(fld):
    for fname in files:
        print(f'{c + 1}. {fname}')
        img = Image.open(f'{FOLDER}/{fname}')
        w, h = img.size
        img = img.crop((TRIM_LEFT, TRIM_TOP, w - TRIM_RIGHT, h - TRIM_BOTTOM))
        # img.show()
        # img = img.filter(ImageFilter.GaussianBlur(20))
        # img = img.filter(ImageFilter.MinFilter(5))
        # img.show()
        w, h = img.size
        new_w = int(w * SIZE_FACTOR)
        new_h = int(h * SIZE_FACTOR)
        new_img = img.resize((new_w, new_h))
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
    return (sub_image < 120).sum() >= 3


def insect_square(arr, arr_gray):
    min_pix = mat_argmin(arr_gray)
    potential_sqr = (min_pix[0] - SQUARE_SIZE[0] // 2, min_pix[1] - SQUARE_SIZE[1] // 2)
    if not is_insect_in_square(arr, arr_gray, potential_sqr):
        return None
    return [potential_sqr[1], potential_sqr[0], potential_sqr[1] + SQUARE_SIZE[1], potential_sqr[0] + SQUARE_SIZE[0]]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


mins = []
maxes = []


def output(P, Q, files):
    c = 0
    for fname in files:
        print(f'{c + 1}. {fname}')
        img = Image.open(f'{FOLDER}/{fname}')
        w, h = img.size
        img = img.crop((TRIM_LEFT, TRIM_TOP, w - TRIM_RIGHT, h - TRIM_BOTTOM))
        w, h = img.size
        new_w = int(w * SIZE_FACTOR)
        new_h = int(h * SIZE_FACTOR)
        new_img = img.resize((new_w, new_h))
        arr = np.asarray(new_img).reshape(-1)
        res_np = get_orthogonal_component(P, Q, arr).reshape((new_h, new_w, 3))
        # print(f'\tMIN: {res_np.min()}, MAX: {res_np.max()}'); mins.append(res_np.min()); maxes.append(res_np.max());
        # res_np -= res_np.min()
        # res_np = (res_np / res_np.max()) * 255
        res_np = APPLY_TO_RESULT(res_np)
        res_np = res_np.astype('uint8')
        gray_np = rgb2gray(res_np)
        square = insect_square(res_np, gray_np)
        sqr_img = new_img.copy()
        if square is not None:
            d = ImageDraw.Draw(sqr_img)
            d.rectangle(square, outline='red', width=3)
            print('\tFound insect!')
        sqr_img.save(f'{THRESHOLD_OUT}/{fname}')
        res_img = Image.fromarray(res_np)
        res_img.save(f'{OUTPUT_FOLDER}/{fname}')
        gray_image = res_img.convert('L')
        # gray_image = Image.fromarray(gray_np)
        gray_image.save(f'{GRAYCSCALE_OUT}/{fname}')

        res_img.close()
        sqr_img.close()
        gray_image.close()
        img.close()
        new_img.close()
        c += 1
        # if 0 < num_images <= c:
        #     break


def clean(force=False):
    if not force:
        confirm = input('Clean (Y/N)? ')
        if confirm.lower() != 'y':
            print('Canceled!')
            return False
    for fld in [OUTPUT_FOLDER, GRAYCSCALE_OUT, THRESHOLD_OUT]:
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
    all_files = os.listdir(FOLDER)
    for i in range(0, len(all_files), NUM_IMAGES_FOR_SVD):
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
    print(f'Done!')


if __name__ == '__main__':
    main()
