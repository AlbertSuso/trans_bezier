# Starting code
import pandas as pd


def check_similar_users(user_id_x, user_id_y, ratings):
    if user_id_x == user_id_y:
        return False

    joint_ratings = pd.merge(
        ratings[ratings["user id"] == user_id_x],
        ratings[ratings["user id"] == user_id_y],
        on="item id",
    )
    if joint_ratings.empty:
        return False

    joint_ratings["rating_diff"] = abs(
        joint_ratings["rating_x"] - joint_ratings["rating_y"]
    )
    if max(joint_ratings["rating_diff"]) <= 1:
        return True
    return False


def get_recommendations(users, movies, ratings, full_name, method, year):
    if users[users["full name"] == full_name].shape[0] == 0:
        return ""

    obj_id = users[users['full name'] == full_name]['user id']
    user_seen = ratings[ratings['user id'] == obj_id[0]]['item id']

    valid_movies_id = movies[movies['release year'] == year]
    # valid_movies_id = valid_movies_id[~valid_movies_id['movie id'].isin(user_seen)]
    valid_movies_id = valid_movies_id['movie id']


    if valid_movies_id.shape[0] == 0:
        return ""


    if method == "by_popularity":
        valid_ratings = ratings[ratings['item id'].isin(valid_movies_id)]
        winer_id = valid_ratings.groupby(['item id']).agg(['count']).idxmax()
        return movies[movies['movie id'] == winer_id[0]]['movie title'][0]

    elif method == "by_rating":
        valid_ratings = ratings[ratings['item id'].isin(valid_movies_id)]
        winer_id = valid_ratings.groupby(['item id']).agg(['avg']).idxmax()
        return movies[movies['movie id'] == winer_id[0]]['movie title'][0]

    elif method == "by_similar_users":
        return ""

    else:
        return ""




if __name__ == '__main__':
    movies = pd.read_csv("/home/albert/Descargas/movies.csv", sep='|')
    ratings = pd.read_csv("/home/albert/Descargas/ratings.csv", sep='|')
    users = pd.read_csv("/home/albert/Descargas/users.csv", sep='|')

    a = get_recommendations(users, movies, ratings, "Ryan James", "by_popularity", 1995)
    print(a)

    print("ENSERIO???")
    id_user = users[users['full name'] == "Ryan James"]['user id'][0]
    recommended_movie_id = movies[movies['movie title'] == a]['movie id'][0]

    filt_ratings = ratings[ratings['user id'] == id_user]
    filt_ratings = filt_ratings[filt_ratings['item id'] == recommended_movie_id]

    print(filt_ratings)

    print("\n\nTOMA YA\n")
    print(movies.head(1))
    print(users.head(1))




