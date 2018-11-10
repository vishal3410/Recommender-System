import sklearn
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.sparse as sp

from sklearn import metrics
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data():
    rating_column = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('../input/ml-100k/u.data', sep='\t',
                          names=rating_column, encoding='latin-1')
    ratings_train = pd.read_csv('../input/ml-100k/ua.base', sep='\t',
                                names=rating_column, encoding='latin-1')
    ratings_test = pd.read_csv('../input/ml-100k/ua.test', sep='\t', 
                          names=rating_column, encoding='latin-1')
    return ratings, ratings_train, ratings_test

def create_matrix(ratings, train, test):
    unique_users = ratings.user_id.unique().shape[0]
    unique_items = ratings.movie_id.unique().shape[0]

    train_matrix = np.zeros((unique_users, unique_items))
    for line in train.itertuples():
        train_matrix[line[1]-1, line[2]-1] = line[3]

    test_matrix = np.zeros((unique_users, unique_items))
    for line in test.itertuples():
        test_matrix[line[1]-1, line[2]-1] = line[3]
        
    return train_matrix, test_matrix

def predict(data, similarity, type='user'):
    if type == 'user':
        mean_user_rating = data.mean(axis=1)
        ratings_diff = (data - mean_user_rating[:, np.newaxis])
        prediction = (mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T)
        
    if type == 'item':
        prediction = data.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])    
    return prediction

def predict_SVD(data):
    u, s, vt = svds(data, k=20)
    s_diagonal = np.diag(s)
    prediction = np.dot(np.dot(u, s_diagonal), vt)
    return prediction

def rmse(prediction, actual_value):
    prediction = prediction[actual_value.nonzero()].flatten()
    actual_value = actual_value[actual_value.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, actual_value))

data, data_train, data_test = load_data()
train_data_matrix, test_data_matrix = create_matrix(data, data_train, data_test)

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_prediction = predict(train_data_matrix, item_similarity, type='item')
prediction_SVD = predict_SVD(train_data_matrix)

user_rmse = rmse(user_prediction, test_data_matrix)
item_rmse = rmse(item_prediction, test_data_matrix)
svd_rmse = rmse(prediction_SVD, test_data_matrix)

print("User based CF RMSE: ", user_rmse)
print("Item based CF RMSE: ", item_rmse)
print("SVD based CF RMSE: ", svd_rmse)
