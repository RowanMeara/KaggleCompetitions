import getdata
import time
import numpy as np
import xgboost as xgb
from sklearn import preprocessing


class Prudential(object):
    def __init__(self):
        self.CATEGORICAL = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6',
                            'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',
                            'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5',
                            'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2',
                            'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8',
                            'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3',
                            'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7',
                            'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12',
                            'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17',
                            'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
                            'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26',
                            'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30',
                            'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35',
                            'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
                            'Medical_History_40', 'Medical_History_41']
        self.CONTINUOUS = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',
                           'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3',
                           'Family_Hist_4', 'Family_Hist_5']

    @staticmethod
    def get_params():
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.05
        params["min_child_weight"] = 240
        params["subsample"] = 1
        params["colsample_bytree"] = 0.67
        params["silent"] = 1
        params["max_depth"] = 7
        plst = list(params.items())
        return plst

    def encode_features(self, xtr, xte, header):
        # Encode Categorical Features
        print('Encoding Features...')
        (cat, cont, other) = getdata.get_feature_indices(self.CATEGORICAL, self.CONTINUOUS, header)
        xtr_cat = xtr[:, cat]
        xtr_cont = xtr[:, cont]
        xte_cat = xte[:, cat]
        xte_cont = xte[:, cont]
        enc = preprocessing.OneHotEncoder(sparse=False)
        enc.fit(np.concatenate((xtr_cat, xte_cat)))
        xtr_cat = enc.transform(xtr_cat)
        xte_cat = enc.transform(xte_cat)

        # Remove features with low counts included during transform
        t = np.sum(xtr_cat, axis=0)
        features_to_remove = []
        for i in range(0, t.size):
            if t[i] < 10:
                features_to_remove.append(i)
        remove = np.array(features_to_remove)
        xtr_cat = np.delete(xtr_cat, remove, axis=1)
        xte_cat = np.delete(xte_cat, remove, axis=1)

        # Encode Continuous Features
        quad_enc = preprocessing.PolynomialFeatures(2)
        xtr_cont = quad_enc.fit_transform(xtr_cont)
        xte_cont = quad_enc.transform(xte_cont)

        # Recombine categorical, continuous, and other features
        xtr = np.concatenate((xtr[:, other], xtr_cont, xtr_cat), axis=1)
        xte = np.concatenate((xte[:, other], xte_cont, xte_cat), axis=1)
        return xtr, xte


if __name__ == "__main__":
    print('Importing data...')
    (xTr, yTr, head) = getdata.import_training('train.csv')
    (xTe, ids) = getdata.import_testing('test.csv')
    model = Prudential()
    # Remove id and response field
    head.pop(0)
    head.pop()
    (xTr, xTe) = model.encode_features(xTr, xTe, head)

    # Put data in xgb format
    (num_samples, d) = xTr.shape
    xg_train = xgb.DMatrix(xTr[0:(num_samples/4), :], yTr[0:num_samples/4])
    xg_eval = xgb.DMatrix(xTr[(num_samples/4):num_samples, :], label=yTr[(num_samples/4):num_samples])
    params = model.get_params()
    eval_list = [(xgtrain, 'train')]
    print("final training size", xTr.shape)

    # Create model
    print('Creating model...')
    start = time.time()
    xgb_model = xgb.train(params, xgtrain, 700)
    end = time.time()

    print("Total Training Time:", (end-start))
    xgb_model.save_model('0001.model')

    # Predictions
    print('Predicting new values...')
    xgtest = xgb.DMatrix(xTe)
    preds = xgb_model.predict(xgtest)
    print(preds.shape)
    for i in range(0, preds.size):
        if preds[i] < 1:
            preds[i] = 1
        elif preds[i] < 8:
            preds[i] = 8
    getdata.create_submission(ids, preds)

