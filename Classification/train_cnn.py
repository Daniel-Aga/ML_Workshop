import numpy as np
import os
import cv2 as cv
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from math import ceil
import matplotlib.pyplot as plt
import random

### INPUT:
TRAIN_FOLDER = 'Labeled/train'  # The folder containing the training samples.
VALID_FOLDER = 'Labeled/valid'  # The folder containing the validation samples.
DATA_SUBFOLDER = 'transformed'  # The subfolder containing the transformed images (64x64 pixels)
TARGET_SUBFOLDER = 'class_txts'  # The subfolder containing the targets (class of each image)
CLASSES_FILE = 'Labeled/classes.txt'  # The file containing the classes names.

### OUTPUT:
OUTPUT_MODEL = 'model.pt'  # The file to save the output model to.

### CONSTANTS:
IMAGES_EXT = 'JPG'
TXT_EXT = 'txt'

### GENERAL:
SHOW_RESULTS_EVERY_EPOCHS = 1  # The number of epochs between each 'print'.

### CNN Metadata:
INPUT_SHAPE = (3, 64, 64)  # The shape of the CNN input.
OUTPUT_SIZE = 6  # The shape of the CNN output.
NUM_TRAIN_SAMPLES = 894
NUM_TEST_SAMPLES = 191


### CNN Training:
EPOCHS = 100
BATCH_SIZE = 100
STD = 0.001  # The STD used for initialization.


### CNN Architecture
NUM_MAX_POOL = 2  # The number of MaxPool layers.
MAX_POOLING_RATIO = 2  # The ratio of MaxPool layers.
CONV_DEFAULT_CHANNELS = (64, 32, 16)  # The number of channels in each convolutional layer.
CONVOLUTION_FILTER_SIZE = (3, 3)  # The size of the convolutional kernels.
CONVOLUTION_FILTER_STRIDE = 1  # The stride of the convolutional kernels.
DROPOUT = 0.3  # The probability of dropout.


train_samples = None
train_labels = None
test_samples = None
test_labels = None


def get_filenames(folder):
    """
    Returns the filenames (without extension) of the files in the given folder, in a shuffled order.
    :param folder: The folder to consider.
    :return: A list of filenames.
    """
    filenames = os.listdir(folder)
    for i in range(len(filenames)):
        filenames[i] = os.path.splitext(filenames[i])[0]
    random.shuffle(filenames)
    return filenames


def get_classes():
    """
    Returns a list containing the insect classes, with added spaces so they are the same length.
    """
    with open(CLASSES_FILE, 'r') as f:
        classes = f.readlines()
    classes = [x[3:].strip() for x in classes]
    max_len = max(map(len, classes))
    classes = [x.ljust(max_len) for x in classes]
    return classes


train_names = get_filenames(f'{TRAIN_FOLDER}/{DATA_SUBFOLDER}')
valid_names = get_filenames(f'{VALID_FOLDER}/{DATA_SUBFOLDER}')


class InsectDataset(Dataset):
    """
    Represents a dataset of insects.
    """
    def __init__(self, train):
        """
        Loads the images and targets and prepares them for training.
        :param train: True if should use the training dataset; False for validation.
        """
        if train:
            names = train_names
            working_folder = TRAIN_FOLDER
        else:
            names = valid_names
            working_folder = VALID_FOLDER
        data_lst = []
        target_lst = []
        for name in names:
            data_name = f'{working_folder}/{DATA_SUBFOLDER}/{name}.{IMAGES_EXT}'
            target_name = f'{working_folder}/{TARGET_SUBFOLDER}/{name}.{TXT_EXT}'
            data_lst.append(cv.imread(data_name))
            with open(target_name, 'r') as f:
                target_lst.append(int(f.read().strip()))

        self.data = np.stack(data_lst).astype(np.float32)
        self.data = np.moveaxis(self.data, -1, 1)
        self.target = np.array(target_lst).astype(np.int64)

        print(f'self.data.shape: {self.data.shape}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, ...], self.target[idx]


class ConvolutionNeuralNetwork(nn.Module):
    """
    Represents a CNN.
    """
    def __init__(self, channels=CONV_DEFAULT_CHANNELS, dropout=None):
        """
        Prepares the CNN according to the parameters given and the global constants.
        :param channels: A tuple containing the channels in each convolutional layer.
        :param dropout: The dropout probability, None for no dropout.
        """
        super(ConvolutionNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        prev_channels = INPUT_SHAPE[0]
        prev_size = INPUT_SHAPE[1]
        conv_layers = []
        i = NUM_MAX_POOL
        for c in channels:
            conv_layers += [
                nn.Conv2d(prev_channels, c, CONVOLUTION_FILTER_SIZE, stride=CONVOLUTION_FILTER_STRIDE),
                nn.ReLU()
            ]
            prev_size -= (CONVOLUTION_FILTER_SIZE[0] - 1)
            if i > 0:
                conv_layers += [
                    nn.MaxPool2d((MAX_POOLING_RATIO, MAX_POOLING_RATIO), stride=MAX_POOLING_RATIO, ceil_mode=True)
                ]
                prev_size = ceil(prev_size / MAX_POOLING_RATIO)
                i -= 1
            if dropout is not None:
                conv_layers.append(nn.Dropout(dropout))
            prev_channels = c
        self.conv_stack = nn.Sequential(*conv_layers)
        self.fc_layer = nn.Linear(prev_size ** 2 * prev_channels, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.fc_layer(x)
        return logits


def train_loop(data_loader, model, loss_fn, optimizer, device):
    """
    Runs one training epoch on the given data.
    Returns the train accuracy and loss.
    """
    for batch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return test_loop(data_loader, model, loss_fn, device)  # Use the function below  (on train data)


def test_loop(data_loader, model, loss_fn, device):
    """
    Runs the model on the given test data.
    Returns the test accuracy and loss.
    Does not modify the model itself.
    """
    size = len(data_loader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        model.eval()
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            # Prediction and loss:
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    return correct, test_loss


def create_model(std, device, channels=CONV_DEFAULT_CHANNELS, dropout=None, xavier=False):
    """
    Creates and returns a neural network model according to the given parameters.
    :param std: standard deviation used for initialization.
    :param device: pytorch device to send the model to (e.g. 'cuda').
    :param channels: convolution network channels structure.
    :param dropout: probability of dropout in dropout layers (no dropout layers if None).
    :param xavier: True if xavier initialization
    """
    model = ConvolutionNeuralNetwork(dropout=dropout, channels=channels)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if xavier:
                torch.nn.init.xavier_normal_(m.weight)
            else:
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            torch.nn.init.normal_(m.bias, mean=0.0, std=std)

    model.apply(init_weights)
    model = model.to(device)
    return model


def create_optimizer(model, learning_rate=None, momentum=None, adam=False, weight_decay=0):
    """
    Creates and returns an optimizer according to the given parameters.
    :param model: The model to optimize.
    :param learning_rate: The learning rate of the optimizer (ignored if adam=True).
    :param momentum: The momentum of the optimizer (ignored if adam=True).
    :param adam: True if Adam optimizer (with default parameters). False if SGD.
    :param weight_decay: The weight decay of the optimizer (ignored if adam=True).
    """
    if adam:
        return torch.optim.Adam(model.parameters())
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def plot_model_results(train_loader, test_loader, model, optimizer, loss_fn, epochs, device,
                       figure_number_0=None,
                       figure_number_1=None, plot_accuracy_and_loss_on_same_figure=True, plt_show=True,
                       random_colors=False, label_prefix=''):
    """
    Trains the given model on the data and plots (and prints) the results.
    Returns the final accuracy and loss.
    :param train_loader: The train data.
    :param test_loader: The test data.
    :param model: The model to train (feed forward / convolution network).
    :param optimizer: The optimizer to use while training.
    :param loss_fn: the loss function for the model
    :param epochs: the number of epochs
    :param figure_number_*: Figure number to plot the results to. If None, will create a new figure.
    :param plot_accuracy_and_loss_on_same_figure: If True, use the same figure for both accuracy and loss (figure_number_0). If False, Accuracy on figure_number_0 and loss on figure_number_1.
    :param plt_show: If true, will call plt.show() after plotting.
    :param random_colors: Randomize the colors on the graph (If False, will use pre-selected colors).
    :param label_prefix: Add to this to the legend before the type of plot.
    """
    epoch_train_accuracies = []
    epoch_train_losses = []
    epoch_test_accuracies = []
    epoch_test_losses = []

    for t in range(epochs):
        train_accuracy, train_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
        test_accuracy, test_loss = test_loop(test_loader, model, loss_fn, device)
        epoch_train_accuracies.append(train_accuracy)
        epoch_train_losses.append(train_loss)
        epoch_test_accuracies.append(test_accuracy)
        epoch_test_losses.append(test_loss)

        if t % SHOW_RESULTS_EVERY_EPOCHS == 0:
            print(
                f'Epoch {t} results: Train Accuracy = {train_accuracy}, Train Loss = {train_loss}, Test Accuracy = {test_accuracy}, Test Loss = {test_loss}')


    # Remove Epoch 0 Results
    del epoch_train_accuracies[0]
    del epoch_train_losses[0]
    del epoch_test_accuracies[0]
    del epoch_test_losses[0]
    epochs -= 1

    if figure_number_0 is None:
        plt.figure()
    else:
        plt.figure(figure_number_0)
    if plot_accuracy_and_loss_on_same_figure:
        plt.title('Accuracy/Loss')
    else:
        plt.title('Accuracy')
    plt.xlabel('Epoch')

    train_accuracy_color = 'green' if not random_colors else np.random.rand(3, )
    test_accuracy_color = 'blue' if not random_colors else np.random.rand(3, )
    train_loss_color = 'red' if not random_colors else train_accuracy_color
    test_loss_color = 'yellow' if not random_colors else test_accuracy_color

    plt.plot(range(epochs), epoch_train_accuracies, color=train_accuracy_color,
             label=(label_prefix + '. ' if label_prefix != '' else '') + 'Train Accuracy')
    plt.plot(range(epochs), epoch_test_accuracies, color=test_accuracy_color,
             label=(label_prefix + '. ' if label_prefix != '' else '') + 'Test Accuracy')
    if not plot_accuracy_and_loss_on_same_figure:
        plt.legend(loc='lower left')
        if plt_show:
            plt.show()
        if figure_number_1 is None:
            plt.figure()
        else:
            plt.figure(figure_number_1)
        plt.xlabel('Epoch')
        plt.title('Loss')
    plt.plot(range(epochs), epoch_train_losses, color=train_loss_color,
             label=(label_prefix + '. ' if label_prefix != '' else '') + 'Train Loss')
    plt.plot(range(epochs), epoch_test_losses, color=test_loss_color,
             label=(label_prefix + '. ' if label_prefix != '' else '') + 'Test Loss')
    plt.legend(loc='lower left')
    if plt_show:
        plt.show()
    print(f'Model Accuracy: {epoch_test_accuracies[-1]}, Model Loss: {epoch_test_losses[-1]}')
    return epoch_test_accuracies[-1], epoch_test_losses[-1]


def get_confusion_matrix(data_loader, model, device):
    """
    Computes the confusion matrix of the given trained model on the given dataset.
    :param data_loader: A DataLoader containing the dataset.
    :param model: The trained model.
    :return: A np matrix where [i,j] index is the number of times the ground truth was 'i' and prediction was 'j'.
    """
    ans = torch.zeros((OUTPUT_SIZE, OUTPUT_SIZE))
    with torch.no_grad():
        model.eval()
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            n = y.shape[0]

            # Prediction and loss:
            pred = model(X)

            chosen = pred.max(axis=1).indices
            for i in range(n):
                ans[y[i], chosen[i]] += 1
            pass

    return ans


def cnn():
    """
    Performs the training, and saves the model.
    :return: None
    """
    global train_samples, train_labels, test_samples, test_labels

    epochs = EPOCHS
    batch_size = BATCH_SIZE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_samples = train_samples.to(device)
    train_labels = train_labels.to(device)
    test_samples = test_samples.to(device)
    test_labels = test_labels.to(device)

    train_loader = torch.utils.data.DataLoader(list(zip(train_samples, train_labels)), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(list(zip(test_samples, test_labels)), batch_size=batch_size, shuffle=True)

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    std = STD

    # Create the model:
    model = create_model(std, device, xavier=True, dropout=DROPOUT)

    print(model)

    # Create the optimizer:
    optimizer = create_optimizer(model, adam=True)

    # Run the model:
    plot_model_results(train_loader, test_loader, model, optimizer, loss_fn, epochs, device,
                       plot_accuracy_and_loss_on_same_figure=False)

    # Print the confusion matrix:
    confusion_matrix = get_confusion_matrix(test_loader, model, device)
    with open(CLASSES_FILE, 'r') as f:
        classes = f.readlines()
        classes = [x[3:].strip() for x in classes]
        max_len = max(map(len, classes))
        classes = [x.ljust(max_len) for x in classes]
    print(' ' * (3 + max_len) + '\t' * 2 + ('\t' * 2).join([f'y={c}'.ljust(2 + max_len) for c in classes]))
    for i in range(OUTPUT_SIZE):
        print(f'pr={classes[i]}'.ljust(3 + max_len) + '\t' * 2 + ('\t' * 2).join(
            [f'{confusion_matrix[j][i]}'.ljust(2 + max_len) for j in range(OUTPUT_SIZE)]))

    # Save the model:
    torch.save(model.state_dict(), OUTPUT_MODEL)

    print("Done!")


def create_datasets():
    """
    Initializes the training and validation datasets.
    :return: None
    """
    global train_samples, train_labels, test_samples, test_labels
    train_dataset = InsectDataset(True)
    test_dataset = InsectDataset(False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=NUM_TRAIN_SAMPLES, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=NUM_TEST_SAMPLES, shuffle=True)
    train_samples, train_labels = iter(train_loader).next()
    test_samples, test_labels = iter(test_loader).next()


def main():
    create_datasets()
    cnn()
    pass


if __name__ == '__main__':
    main()
