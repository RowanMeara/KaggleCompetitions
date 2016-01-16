import csv
import numpy as np


def import_training(filename):
    file_reader = csv.reader(open(filename, 'r'))
    data = []
    next(file_reader)
    for row in file_reader:
        data.append(row)
    data = np.matrix(data)
    print("Importing training data...")
    (r, c) = data.shape
    ytr = data[:, c-1]
    # remove id for training as well as result
    xtr = data[:, 1:(c-1)]
    # convert product_info_2 to int
    for i in range(0, r):
        xtr[i, 1] = int(xtr[i, 1], 16)
    xtr[xtr == ''] = 0.0
    xtr = np.array(xtr.astype(np.float))
    ytr = np.array(ytr.astype(np.float)).flatten()
    return xtr, ytr


def import_testing(filename):
    file_reader = csv.reader(open(filename, 'r'))
    data = []
    next(file_reader)
    for row in file_reader:
        data.append(row)
    data = np.matrix(data)
    print("Importing testing data...")

    # remove id for data
    (r, c) = data.shape
    ids = data[:, 0]
    xte = data[:, 1:c]
    # convert product_info_2 to int
    for i in range(0, r):
        xte[i, 1] = int(xte[i, 1], 16)
    xte[xte == ''] = 0.0
    xte = np.array(xte.astype(np.float))
    ids = np.array(ids.astype(np.float)).flatten()
    return xte, ids


def create_submission(ids, preds):
    body = np.matrix([ids, preds]).transpose()
    np.savetxt('submission.csv', body, fmt='%i', delimiter=",", header="Id,Response",comments='')

def get_categorical():
    return np.array([0, 1, 2, 4, 5, 6, 12, 13, 15] + list(range(17, 28)) +
                    [29, 30, 31, 32]+list(range(38, 46)) + [47, 48, 49, 50] +
                    list(range(52, 60)) + list(range(61, 68)) + list(range(69, 78)))

