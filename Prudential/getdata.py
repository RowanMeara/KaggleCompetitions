import csv
import numpy as np


def import_training(filename):
    file_reader = csv.reader(open(filename, 'r'))
    data = []
    header = next(file_reader)
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
    return xtr, ytr, header


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


def get_feature_indices(categorical, continuous, header):
    cat = []
    cont = []
    other = []
    for i in range(0, len(header)):
        if header[i] in categorical:
            cat.append(i)
        elif header[i] in continuous:
            cont.append(i)
        else:
            other.append(i)
    return cat, cont, other
