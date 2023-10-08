import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Build non-personalised recommender system by using Movielen 1M:
# The non-personalised recommender system is based on the average rating of each movie
# The aim is to recommend the movies with the highest average rating to the users

# load data
def load_data():
    # Load data from files
    ratings = pd.read_table("./ml-1m/ratings.dat", sep="::", header=None, names=["UserID", "MovieID", "Rating", "Timestamp"], engine="python", encoding="latin-1")
    df_movies = pd.read_table("./ml-1m/movies.dat", sep="::", header=None, names=["MovieID", "Title", "Genres"], engine="python",encoding="latin-1")
    df_users = pd.read_table("./ml-1m/users.dat", sep="::", header=None, names=["UserID", "Gender","Age","Occupation","Zip-code"], engine="python",encoding="latin-1")
    return ratings, df_movies, df_users

# Get the average rating for each movie
def get_avg_rating(ratings,df_movies):
   
    df_avg_ratings = ratings.groupby("MovieID")["Rating"].mean().round().astype(int).reset_index()

    df_avg_ratings = df_avg_ratings.rename(columns={"Rating": "AvgRating"})

    df_movies = pd.merge(df_movies, df_avg_ratings, on="MovieID", how="left")

    return df_movies

# get the missing movies, the aim is to not recommend these movies.
def get_existing_movies(df_movies):
    # Get the set of existing movies
    existing_movies = set(df_movies["MovieID"].unique())
    missing_movies = list(set(range(1, 3953)) - existing_movies)
    return missing_movies

# get the number of users and items
def get_num_items_users(ratings):
    # Get the number of users and items
    num_items = pd.read_csv("./ml-1m/movies.dat",delimiter="::",header=None,engine='python',encoding="latin-1")[0].max()
    num_users = pd.read_csv("./ml-1m/ratings.dat",delimiter="::",header=None,engine='python',encoding="latin-1")[0].max()
    return num_items,num_users

# get the user-item rating matrix
# nan value means missing value
def get_user_item_ratingMatrix(ratings,num_users,num_items):
    # Create a user-item rating matrix
    user_item_ratingMatrix = np.zeros((num_users, num_items))
    for row in ratings.itertuples():
        user_item_ratingMatrix[row[1]-1, row[2]-1] = row[3]
        
    user_item_ratingMatrix = pd.DataFrame(user_item_ratingMatrix.T, index=range(1, num_items+1), columns=range(1, num_users+1))
    user_item_ratingMatrix = user_item_ratingMatrix.rename_axis("UserID", axis="columns")
    user_item_ratingMatrix = user_item_ratingMatrix.rename_axis("MovieID", axis="rows")
    # put nan as 0
    user_item_ratingMatrix = user_item_ratingMatrix.fillna(0)   
    return user_item_ratingMatrix

# make predictions
def make_prediction(user_id, user_item_ratingMatrix, missing_movies, df_movies):
    # Get the average rating for each movie
    avg_ratings = df_movies.set_index("MovieID")["AvgRating"]
    
    # Get the ratings for the given user
    item_rating = user_item_ratingMatrix.loc[:, user_id]
    
    # Replace missing ratings with the average rating for the movie
    item_rating[item_rating == 0] = avg_ratings[item_rating == 0]
    
    # Return the predicted ratings
    return item_rating

# make recommendations
def make_recommendation(user_id,item_rating,df_movies,num_recommendations=10):
    recommendations = item_rating.sort_values(ascending=False).head(num_recommendations)
    # Get the movie title
    recommendations = pd.merge(recommendations, df_movies[["MovieID", "Title","Genres"]], on="MovieID", how="left")
    recommendations = recommendations.rename(columns={user_id: "Rating"})
    return recommendations
    
# main function about non-personalised recommender system
def non_personalised_rc(user_id=10, num_recommendations=10):
    
    ratings, df_movies, df_users = load_data()
    df_movies = get_avg_rating(ratings,df_movies)
    missing_movies = get_existing_movies(df_movies)
    num_items,num_users = get_num_items_users(ratings)
    user_item_ratingMatrix = get_user_item_ratingMatrix(ratings,num_users,num_items)
    watched_movies = user_item_ratingMatrix[user_item_ratingMatrix[user_id] != 0][user_id]
    item_rating = make_prediction(user_id,user_item_ratingMatrix,missing_movies,df_movies)
    item_rating = item_rating.drop(watched_movies.index)
    recommendations = make_recommendation(user_id,item_rating,df_movies,num_recommendations=num_recommendations)

    return recommendations

# if you want to test this function, you can uncomment the following code
# recommendations = non_personalised_rc(user_id=10, num_recommendations=10)
# print(recommendations)

# Main function about evaluation on non-personalised recommender system
# train_set and test_set are part of the ratings
# Only train_set is used to make predictions, and test_set is used to evaluate the performance
# Revelent score is item ratings
def non_personalised_rmse_nDCG_hitRate_evaluations(k=100):
    
    ratings, df_movies,_ = load_data()
    train_ratings, test_ratings = train_test_split(ratings, test_size = 0.2)
    test_set = test_ratings["UserID"].unique()
    df_avg_ratings = train_ratings.groupby("MovieID")["Rating"].mean().round().astype(int).reset_index()
    df_avg_ratings = df_avg_ratings.rename(columns={"Rating": "AvgRating"})
    
    missing_movies = get_existing_movies(df_movies)
    num_items,num_users = get_num_items_users(ratings)

    # Merge the average rating with the movie dataframe
    train_movies = pd.merge(df_movies, df_avg_ratings, on="MovieID", how="left")
    user_item_ratingMatrix = get_user_item_ratingMatrix(ratings,num_users,num_items)
    
    train_user_item_ratingMatrix = get_user_item_ratingMatrix(train_ratings,num_users,num_items)
    test_user_item_ratingMatrix = get_user_item_ratingMatrix(test_ratings,num_users,num_items)
    
    total_num = len(test_set)
    rmses = []
    dcg_list = []
    idcg_list = []
    hit_rates = []
    for user_id in test_set:
        dcg = 0
        idcg = 0
        
        train_watched_movies = train_user_item_ratingMatrix[train_user_item_ratingMatrix[user_id] != 0][user_id]
        predicted_rating = make_prediction(user_id,user_item_ratingMatrix,missing_movies,train_movies)
        predicted_rating = predicted_rating.drop(train_watched_movies.index)
        predictions = predicted_rating.sort_values(ascending=False).head(k)
        
        df_predictions = pd.DataFrame(predictions).reset_index()
        
        prediction_id = df_predictions.MovieID.values
        
        truth_rating = test_user_item_ratingMatrix[user_id][prediction_id]
        
        actual_rating = test_user_item_ratingMatrix[user_id]
        actual_rating = pd.DataFrame(actual_rating)
        actual_watched_movies = actual_rating[actual_rating[user_id] != 0].index
        actual_ratings = actual_rating[actual_rating[user_id] != 0].values
        
        predicted_ratings = predicted_rating[actual_watched_movies].values
        
        test_watched_movies = test_user_item_ratingMatrix[test_user_item_ratingMatrix[user_id] != 0][user_id].index
        
        num_hits = len(set(prediction_id) & set(test_watched_movies))
        hit_rate = num_hits / len(prediction_id)
        hit_rates.append(hit_rate)

        for i in range(k):
            # Calculate DCG
            dcg += (2**truth_rating.iloc[i] - 1)/np.log2(i+2)
            # Calculate IDCG
            idcg += (2**predictions.iloc[i] - 1)/np.log2(i+2)
            
        dcg_list.append(dcg)
        idcg_list.append(idcg)
        
        error = np.subtract(actual_ratings,predicted_ratings)
        square_error = np.square(error)
        mean_square_error = np.nanmean(square_error)
        rmse = np.sqrt(mean_square_error)
        rmses.append(rmse)
        now = len(rmses)
        process = now/total_num
        print("process: {:.2%}".format(process),end="\r")
        
    final_rmse = np.mean(rmses)
    ndcg = np.mean(np.array(dcg_list)/np.array(idcg_list))
    final_hit_rate = np.mean(hit_rates)
    return final_rmse,ndcg,final_hit_rate
