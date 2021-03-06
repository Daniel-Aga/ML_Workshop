import numpy as np
import os
import cv2 as cv
import torch
from pathlib import Path
from train_cnn import ConvolutionNeuralNetwork, DROPOUT, CONV_DEFAULT_CHANNELS, get_classes

### INPUT:
INPUT_MODEL = 'model.pt'  # The file in which the model is saved.
IMAGES_FOLDER = 'outs_mishmar_cropped'  # The folder in which the unlabeled cropped images are stored.


### OUTPUT:
THRESH_INFERRED = 'inferred_mishmar'  # The output folder.

def infer():
    """
    Uses the model to infer the types of the given insects.
    :return: None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    classes = get_classes()

    for c in classes:
        Path(f'{THRESH_INFERRED}/{c.strip()}').mkdir(parents=True, exist_ok=True)

    model = ConvolutionNeuralNetwork(dropout=DROPOUT, channels=CONV_DEFAULT_CHANNELS)
    model.load_state_dict(torch.load(INPUT_MODEL))
    model = model.to(device)
    model.eval()
    for fname in os.listdir(IMAGES_FOLDER):
        print(fname)
        img_name = f'{IMAGES_FOLDER}/{fname}'
        img = cv.imread(img_name)
        nparr = img.astype(np.float32)
        nparr = np.moveaxis(nparr, -1, 0)

        X = torch.from_numpy(nparr)
        X = X.unsqueeze(0)
        X = X.to(device)

        pred = model(X)

        chosen = pred.max(axis=1).indices[0].item()

        out_name = f'{THRESH_INFERRED}/{classes[chosen].strip()}/{fname}'
        cv.imwrite(out_name, cv.imread(f'{IMAGES_FOLDER}/{fname}'))
    print('Done!')


def main():
    infer()
    pass


if __name__ == '__main__':
    main()