import getdata
import numpy as np
from sklearn import svm, preprocessing, cross_validation


if __name__ == "__main__":
    print('Importing data...')
    (xTr, yTr) = getdata.import_training('train.csv')
    (xTe, ids) = getdata.import_testing('test.csv')

    print('Encoding Features...')
    categorical = getdata.get_categorical()
    enc = preprocessing.OneHotEncoder(categorical_features=categorical, sparse=False)
    enc.fit(xTr)
    (num_features, sample_size) = xTr.shape
    enc.fit(np.concatenate((xTe, xTr)))
    xTr = enc.transform(xTr)
    xTe = enc.transform(xTe)
    # Remove useless features included during transform
    t = np.sum(xTr, axis=0)
    features_to_remove = []
    for i in range(0, t.size):
        if t[i] < 5:
            features_to_remove.append(i)
    remove = np.array(features_to_remove)
    xTr = np.delete(xTr, remove, axis=1)
    xTe = np.delete(xTe, remove, axis=1)


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        xTr, yTr, test_size=0.2, random_state=0)

    print('Creating model...')
    clf = svm.LinearSVC()
    clf.fit(xTr,yTr)

    print('Predicting new values...')
    preds = clf.predict(xTe)
    getdata.create_submission(ids, preds)


