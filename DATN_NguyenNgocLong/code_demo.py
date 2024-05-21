from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Đọc dữ liệu
ratings_df = pd.read_csv("Data/rating.csv")
movies_df = pd.read_csv("Data/movie.csv")
users_df = pd.read_csv("Data/user.csv")

# Merge dữ liệu
movie_ratings_df = pd.merge(ratings_df, movies_df[['MovieID', 'Title', 'Genres']], on='MovieID', how='left')
user_movie_ratings_df = pd.merge(movie_ratings_df, users_df[['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']],
                                 on='UserID', how='left')

# Tạo ma trận đánh giá (UserID x MovieID)
rating_matrix = user_movie_ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating')
rating_matrix = rating_matrix.fillna(0)

# Lọc cộng tác: Sử dụng KNN để tìm người dùng giống nhau
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(rating_matrix.values)

# Lọc dựa trên nội dung (Content-Based Filtering)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
genres_matrix = tfidf_vectorizer.fit_transform(movies_df['Genres'].fillna(''))
content_similarity = linear_kernel(genres_matrix, genres_matrix)

def recommend_movies(user_id):
    _, similar_users = knn_model.kneighbors([rating_matrix.loc[user_id]])
    recommended_movies = rating_matrix.loc[similar_users.flatten()].mean(axis=0).sort_values(ascending=False)
    user_rated_movies = rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index
    final_recommendations = recommended_movies[~recommended_movies.index.isin(user_rated_movies)].index
    return final_recommendations[:10]

def content_based_recommendation(movie_id):
    movie_idx = movies_df[movies_df['MovieID'] == movie_id].index[0]
    similarity_scores = list(enumerate(content_similarity[movie_idx]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_movies = [movies_df.iloc[movie[0]] for movie in sorted_movies[1:11]]
    return recommended_movies

def hybrid_recommendation(user_id, movie_id):
    _, similar_users = knn_model.kneighbors([rating_matrix.loc[user_id]])
    similar_movies = content_similarity[movie_id].argsort()[:-6:-1]
    combined_recommendations = set(similar_users.flatten()) | set(similar_movies)
    user_rated_movies = rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index
    final_recommendations = [movie for movie in combined_recommendations if movie not in user_rated_movies]
    return final_recommendations

@app.route('/')
def index():
    movies = movies_df[['MovieID', 'Title']].to_dict(orient='records')
    users = users_df['UserID'].tolist()
    return render_template('index.html', movies=movies, users=users)

@app.route('/recommend/collaborative', methods=['POST'])
def recommend_collaborative():
    user_id = int(request.form['user_id'])
    recommendations = recommend_movies(user_id)
    collab_movies = movies_df[movies_df['MovieID'].isin(recommendations)][['MovieID', 'Title', 'Genres']]
    return render_template('index.html', collab_movies=collab_movies.to_dict(orient='records'), movies=movies_df[['MovieID', 'Title']].to_dict(orient='records'), users=users_df['UserID'].tolist())

@app.route('/recommend/content', methods=['POST'])
def recommend_content():
    movie_id = int(request.form['movie_id'])
    recommendations = content_based_recommendation(movie_id)
    return render_template('index.html', content_movies=[movie.to_dict() for movie in recommendations], movies=movies_df[['MovieID', 'Title']].to_dict(orient='records'), users=users_df['UserID'].tolist())

@app.route('/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    user_id = int(request.form['hybrid_user_id'])
    movie_id = int(request.form['hybrid_movie_id'])
    recommendations = hybrid_recommendation(user_id, movie_id)
    hybrid_movies = movies_df[movies_df['MovieID'].isin(recommendations)][['MovieID', 'Title', 'Genres']]
    return render_template('index.html', hybrid_movies=hybrid_movies.to_dict(orient='records'), movies=movies_df[['MovieID', 'Title']].to_dict(orient='records'), users=users_df['UserID'].tolist())

if __name__ == '__main__':
    app.run(debug=True)