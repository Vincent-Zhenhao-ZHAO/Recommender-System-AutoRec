import torch
from torch import nn, div, square, norm
from torch.nn import functional as F

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

# Code reference: https://github.com/tuanio/AutoRec
# Paper reference: http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf


# Create user-item rating matrix
# Example: user_item_ratingMatrix[user_id-1][item_id-1] = rating
def create_user_item_ratingMatrix(df,num_users,num_items):
    user_item_ratingMatrix = torch.zeros((num_users, num_items))
    for row in df.itertuples():
        user_item_ratingMatrix[row[1]-1, row[2]-1] = row[3]
    return user_item_ratingMatrix

# Loading data 
# Upath: path to user data
# Mpath: path to movie data
# Rpath: path to rating data
def load_data(Upath,Mpath,Rpath):
    num_users = pd.read_csv(Upath,delimiter="::",header=None,engine='python', encoding='latin-1')[0].max()
    num_items = pd.read_csv(Mpath,delimiter="::",header=None,engine='python',encoding='latin-1')[0].max()
    df_ratings = pd.read_csv(Rpath, sep='::', names=['user_id', 'MovieID', 'rating', 'timestamp'],encoding='latin-1',engine='python')
    df_movies = pd.read_csv(Mpath, sep="::", header=None, names=["MovieID", "Title", "Genres"], engine="python",encoding='latin-1')
    user_item_ratingMatrix = torch.zeros((num_users, num_items))
    user_item_ratingMatrix = create_user_item_ratingMatrix(df_ratings,num_users,num_items)
    return user_item_ratingMatrix, int(num_users), int(num_items), df_ratings, df_movies

# Convert a list of items into a PyTorch LongTensor,
# DataLoaders require a collate_fn to convert a list of items into a batch
def collate_fn(batch):
    return torch.LongTensor(batch)

# turn the whole dataset into a DataLoader
def Create_train_test(num_items):
    
    whole_dl = DataLoader(torch.arange(num_items), shuffle=False,num_workers=0,batch_size=num_items,collate_fn=collate_fn)
    
    return whole_dl

# This function has been commented out, as it is not used in the final version
# def watched_movies(user_id, user_item_ratingMatrix):
#     user_item_ratingMatrix = user_item_ratingMatrix.numpy()
#     df_user_item_ratingMatrix = pd.DataFrame(user_item_ratingMatrix)
#     return df_user_item_ratingMatrix[user_id].to_numpy().nonzero()[0]

# AutoRec model
# Reference: https://github.com/tuanio/AutoRec
# Paper reference: http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf
class AutoRec(nn.Module):
    def __init__(self, visibleDimensions, hiddenDimensions, learningRate):
        super().__init__()
        self.learningRate = learningRate
        self.weight1 = nn.Parameter(torch.randn(visibleDimensions, hiddenDimensions))
        self.weight2 = nn.Parameter(torch.randn(hiddenDimensions, visibleDimensions))
        self.bias1 = nn.Parameter(torch.randn(hiddenDimensions))
        self.bias2 = nn.Parameter(torch.randn(visibleDimensions))
    
    def regularization(self):
        return div(self.learningRate, 2) * (square(norm(self.weight1)) + square(norm(self.weight2)))
    
    def forward(self, data):
        encoder = self.weight2.matmul(data.T).T + self.bias1
        return self.weight1.matmul(encoder.sigmoid().T).T + self.bias2

# Training model
# Code Reference: https://github.com/tuanio/AutoRec
# Difference from the original code: make 0 rating as -1, then put into the training, and make -1 rating as 0
# Aim: to avoid the loss of 0 rating
# Paper reference: http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf
def training(model,train_set,user_item_ratingMatrix,optimizer,criterion,device):
    lossList = []
    for _ , item_idx in enumerate(train_set):
        ratings = user_item_ratingMatrix[:,item_idx].squeeze().permute(1,0).to(device)
        ratings[ratings==0] = -1
        predict_ratings = model(ratings)
        ratings[ratings==-1] = 0
        loss = criterion(ratings, predict_ratings * torch.sign(ratings)) + model.regularization()       
        lossList.append(loss.item())        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return lossList

# Rating prediction
# Code Reference: https://github.com/tuanio/AutoRec
# rating prediction range from 1 to 5, as minimum rating is 1 and maximum rating is 5
def predict_ratings(model ,user_item_ratingMatrix, whole_set):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for _, items_idx in enumerate(whole_set):
            ratings = user_item_ratingMatrix[:, items_idx].squeeze().permute(1, 0).to(device)
            ratings[ratings == 0] = -1
            ratings_prediction = model(ratings) * torch.sign(ratings)
            ratings_prediction[ratings == -1] = 0
            ratings_prediction = torch.ceil(ratings_prediction)
            ratings_prediction = torch.clamp(ratings_prediction, 1, 5)
            break
    return ratings_prediction

# Get recommendations
# This function has been commented out, as it is not used in the final version
# Output: top k movies
# def getRecommendations(user_id, df_movies,num_recommendations=5,ratings_prediction=None):
#     top_k_indices = np.argsort(-ratings_prediction, axis=1)[:, :num_recommendations]
#     top_k_movies = top_k_indices[user_id-1]
#     top_k_movies = [movie_id + 1 for movie_id in top_k_movies]
    
#     top_k_movies = df_movies[df_movies['MovieID'].isin(top_k_movies)]
#     return top_k_movies

# Main function for recommendation
# watched_mask: mask for movies that user has watched
def cf_personalised(user_id=100, num_recommendations=5):
    # The user_id in the dataset starts from 1, so we need to minus 1
    # in the dataframe it starts from 0
    user_id = user_id - 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    user_item_ratingMatrix, num_users, num_items, df_ratings, df_movies = load_data(Upath="./ml-1m/users.dat",Mpath="./ml-1m/movies.dat",Rpath="./ml-1m/ratings.dat")
    whole_set = Create_train_test(num_items=num_items)
    autoRec = AutoRec(visibleDimensions=num_users, hiddenDimensions=500, learningRate=0.0001).to(device)
    autoRec.load_state_dict(torch.load('./model/AutoRec.pth'))
    autoRec.eval()
    predictions = predict_ratings(autoRec, user_item_ratingMatrix, whole_set)
    df_user_item_ratingMatrix = pd.DataFrame(user_item_ratingMatrix.T.numpy())
    watched_mask = np.where(user_item_ratingMatrix != 0, 1, 0)
    ratings_prediction = np.where(watched_mask == 0, predictions.T, user_item_ratingMatrix)
    df_ratings_prediction = pd.DataFrame(ratings_prediction.T)
    watched_movies = df_user_item_ratingMatrix[user_id][df_user_item_ratingMatrix[user_id] != 0]
    user_ratings_predictions = df_ratings_prediction[user_id].drop(watched_movies.index)
    user_rating_prediction = user_ratings_predictions.sort_values(ascending=False)[:num_recommendations+1]
    prediciton_movie = user_rating_prediction.index
    top_k_movies = df_movies[df_movies['MovieID'].isin(prediciton_movie)]
    return top_k_movies

# If you want to train the model, please uncomment the following code
# top_k_movies = cf_personalised(user_id=100, num_recommendations=5)
# print(top_k_movies)

# Evaluation function, calculate RMSE
# Code Reference: https://github.com/tuanio/AutoRec
# Apply the same method to make 0 rating as -1, then put into the training, and make -1 rating as 0
# Return RMSE
def eval_epoch(model, test_set, criterion,user_item_ratingMatrix):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    truth = []
    predict = []
    loss = []
    with torch.no_grad():
        for _ ,items_idx in enumerate(test_set):
            ratings = user_item_ratingMatrix[:, items_idx].squeeze().permute(1,0).to(device)
            ratings[ratings==0] = -1
            ratings_prediction = model(ratings)
            ratings[ratings==-1] = 0
            truth.append(ratings)
            predict.append(ratings_prediction * torch.sign(ratings))       
            single_loss = criterion(ratings, ratings_prediction * torch.sign(ratings)) + model.regularization()
            loss.append(single_loss.item())

    rmse = torch.Tensor([torch.sqrt(square(ratings - ratings_prediction).sum() / torch.sign(ratings).sum())
                            for ratings, ratings_prediction in zip(truth, predict)]).mean().item()
    return rmse


# Main function for evaluation RMSE
# test set: 20% of the whole dataset
def rmse_evaluations():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.MSELoss().to(device)
    user_item_ratingMatrix, num_users, num_items, df_ratings, df_movies = load_data(Upath="./ml-1m/users.dat",Mpath="./ml-1m/movies.dat",Rpath="./ml-1m/ratings.dat")
    
    train_items,test_items = train_test_split(torch.arange(num_items),
                                           test_size=0.2,
                                           random_state=12)
    
    test_set = DataLoader(test_items, shuffle=False,num_workers=0,batch_size=num_items,collate_fn=collate_fn)
    
    autoRec = AutoRec(visibleDimensions=num_users, hiddenDimensions=500, learningRate=0.0001).to(device)
    autoRec.load_state_dict(torch.load('./model/AutoRec.pth'))
    autoRec.eval()
    rmse = eval_epoch(autoRec, test_set, criterion, user_item_ratingMatrix=user_item_ratingMatrix)
    return rmse

# nDCG and hit rate evaluation
# Train_set and test_set are from the ratings.csv
# Then find the item id, and do the same thing as the evaluation function
# The aim of this is to help on the evaluation of the recommendation system
# The relevance of the item is the rating of the item
# In the final list, the nan and 0 will be removed
def nDCG_hitRate_evaluations(k=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    user_item_ratingMatrix, num_users, num_items, df_ratings, df_movies = load_data(Upath="./ml-1m/users.dat",Mpath="./ml-1m/movies.dat",Rpath="./ml-1m/ratings.dat")
    train_set, test_set = train_test_split(df_ratings, test_size=0.2, random_state=12) 
    train_dl = train_set["MovieID"].values - 1
    test_dl = test_set["MovieID"].values - 1
    train_dl = DataLoader(train_dl, batch_size=num_items, shuffle=True)
    test_dl = DataLoader(test_dl, batch_size=num_items, shuffle=True)
    train_user_item_ratingMatrix = create_user_item_ratingMatrix(train_set, num_users, num_items)
    test_user_item_ratingMatrix = create_user_item_ratingMatrix(test_set, num_users, num_items)
    autoRec = AutoRec(visibleDimensions=num_users, hiddenDimensions=500, learningRate=0.0001).to(device)
    autoRec.load_state_dict(torch.load('./model/AutoRec.pth'))
    autoRec.eval()
    predictions = predict_ratings(autoRec, train_user_item_ratingMatrix, train_dl)
    df_train_user_item_ratingMatrix = pd.DataFrame(train_user_item_ratingMatrix.T.numpy())
    df_test_user_item_ratingMatrix = pd.DataFrame(test_user_item_ratingMatrix.T.numpy())
    watched_mask = np.where(train_user_item_ratingMatrix != 0, 1, 0)
    ratings_prediction = np.where(watched_mask == 0, predictions.T, train_user_item_ratingMatrix)
    df_ratings_prediction = pd.DataFrame(ratings_prediction.T)
   
    ndcg_list = []
    hit_rates = []
    
    for userID in range(0,num_users):
        
        watched_movies = df_train_user_item_ratingMatrix[userID][df_train_user_item_ratingMatrix[userID] != 0]
        user_ratings_predictions = df_ratings_prediction[userID].drop(watched_movies.index)
        user_rating_prediction = user_ratings_predictions.sort_values(ascending=False)[:k]
        top_k_ratings = user_rating_prediction.values
        prediciton_movie = user_rating_prediction.index
        truth_rating = df_test_user_item_ratingMatrix[userID][prediciton_movie].values
        user_watched_movie = df_test_user_item_ratingMatrix[userID].to_numpy().nonzero()[0]
        num_hits = len(set(prediciton_movie) & set(user_watched_movie))
        hit_rate = num_hits / k
        hit_rates.append(hit_rate)
        dcg = 0
        idcg = 0
        for i in range(k):
            dcg += (2**truth_rating[i] - 1) / np.log2(i+2)
            idcg += (2**top_k_ratings[i] - 1) / np.log2(i+2)
        if dcg == 0 or idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / idcg
        ndcg_list.append(ndcg)
        process = len(ndcg_list) / num_users
        print("process: {:.2%}".format(process),end="\r")
        
    ndcg_list = [x for x in ndcg_list if x >= 0.01]
    hit_rates = [x for x in hit_rates if x != 0]

    ndcg = np.mean(ndcg_list)
    hit_rate = np.mean(hit_rates)
    return ndcg, hit_rate

