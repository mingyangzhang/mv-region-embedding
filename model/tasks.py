import json

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

crime = np.load("../data/crime_counts.npy")[:, 0]
cd = json.load(open("../data/mh_cd.json"))
cd_labels = np.zeros((180))
for i in range(180):
    cd_labels[i] = cd[str(i)]

def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    return y_pred

def kf_predict(X, Y):

    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def compute_metrics(y_pred, y_test):
    y_pred[y_pred<0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2

def predict_crime(emb):
    y_pred, y_test = kf_predict(emb, crime)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)

    print("MAE: ", mae)
    print("RMSE: ", rmse)
    print("R2: ", r2)
    return mae, rmse, r2

def lu_classify(emb):

    n = 12
    kmeans = KMeans(n_clusters=n, random_state=3)
    emb_labels = kmeans.fit_predict(emb)

    nmi = normalized_mutual_info_score(cd_labels, emb_labels)
    print("emb nmi: {:.3f}".format(nmi))
    ars = adjusted_rand_score(cd_labels, emb_labels)
    print("emb ars: {:.3f}".format(ars))
