import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Đọc dữ liệu
ratings_df = pd.read_csv("Data/rating.csv")
movies_df = pd.read_csv("Data/movie.csv")
users_df = pd.read_csv("Data/user.csv")

# Merge dữ liệu
movie_ratings_df = pd.merge(ratings_df, movies_df[['MovieID', 'Title', 'Genres']], on='MovieID', how='left')
user_movie_ratings_df = pd.merge(movie_ratings_df, users_df[['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']],
                                 on='UserID', how='left')
print(user_movie_ratings_df)
# Tạo ma trận đánh giá (UserID x MovieID)
rating_matrix = user_movie_ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating')
print(rating_matrix)
# Xử lý giá trị thiếu
rating_matrix = rating_matrix.fillna(0)

# Lọc cộng tác: Sử dụng KNN để tìm người dùng giống nhau
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(rating_matrix.values)

#Lọc cộng tác(Collaborative Filtering)

def recommend_movies(user_id):
    if user_id not in rating_matrix.index:
        print(f"User {user_id} not found in the dataset.")
        return []

    _, similar_users = knn_model.kneighbors([rating_matrix.loc[user_id]])

    # Kiểm tra xem có người dùng giống nhau hay không
    if len(similar_users) == 0 or similar_users.flatten()[0] not in rating_matrix.index:
        print(f"No similar users found for user {user_id}.")
        return []

    # Lấy danh sách phim mà những người dùng giống nhau đã đánh giá cao
    recommended_movies = rating_matrix.loc[similar_users.flatten()].mean(axis=0).sort_values(ascending=False)

    # Lọc bỏ các phim đã được người dùng đánh giá
    user_rated_movies = rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index
    final_recommendations = recommended_movies[recommended_movies.index.isin(user_rated_movies)].index

    return final_recommendations


# Thực hiện gợi ý cho một người dùng
user_id_to_recommend =int(input("Nhập người dùng cân gợi ý phim: "))
recommendations = recommend_movies(user_id_to_recommend)


# Hiển thị các phim được gợi ý
recommended_movies = movies_df[movies_df['MovieID'].isin(recommendations)][['MovieID', 'Title', 'Genres']]
print(recommended_movies.to_string())

print("\nDự đoán của mô hình:")
predicted_ratings = pd.Series([5] * len(recommendations), index=recommendations)
print(predicted_ratings)

#Lọc dựa trên nội dung ( Content-Based Filtering)

# Tính toán vector đặc trưng của mỗi phim dựa trên thông tin nội dung (Genres)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
genres_matrix = tfidf_vectorizer.fit_transform(movies_df['Genres'].fillna(''))

# Tính toán độ tương tự cosine giữa các phim dựa trên vector đặc trưng
content_similarity = linear_kernel(genres_matrix, genres_matrix)

# Hàm gợi ý dựa trên Content-Based Filtering
def content_based_recommendation(movie_id):
    # Lấy index của phim trong DataFrame movies_df
    movie_idx = movies_df[movies_df['MovieID'] == movie_id].index[0]

    # Tính toán độ tương tự cosine giữa phim này và tất cả các phim khác
    similarity_scores = list(enumerate(content_similarity[movie_idx]))

    # Sắp xếp các phim theo độ tương tự giảm dần
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Lấy danh sách các phim được gợi ý (loại bỏ phim đầu tiên vì đó là chính phim đang xét)
    recommended_movies = [movies_df.iloc[movie[0]] for movie in sorted_movies[1:11]]

    return recommended_movies

# Thực hiện gợi ý cho một phim
movie_id_to_recommend = int(input("Nhập phim cần gợi ý ( lọc theo nội dung): "))
content_based_recommendations = content_based_recommendation(movie_id_to_recommend)

# Hiển thị các phim được gợi ý
print("Các phim được gợi ý dựa trên Content-Based Filtering:")
for movie in content_based_recommendations:
    print(f"{movie['Title']} - Genres: {movie['Genres']}")



# Tính toán độ tương tự cosine giữa các phim dựa trên vector đặc trưng
content_similarity = linear_kernel(genres_matrix, genres_matrix)

# In ra ma trận độ tương tự cosine
print("Ma trận độ tương tự cosine:")
#np.set_printoptions(threshold=np.inf)  # Đảm bảo in ra tất cả giá trị trong mảng
print(content_similarity)
# In ra đánh giá của người dùng
print("Đánh giá của người dùng:")
print(rating_matrix)


# Hàm dự đoán dựa trên lọc cộng tác và lọc dựa trên nội dung
def hybrid_recommendation(user_id, movie_id):
    # Lọc cộng tác: Tìm người dùng giống nhất
    _, similar_users = knn_model.kneighbors([rating_matrix.loc[user_id]])

    # Lọc dựa trên nội dung: Tìm phim giống nhất
    similar_movies = content_similarity[movie_id].argsort()[:-6:-1]

    # Kết hợp các gợi ý từ cả hai phương pháp
    combined_recommendations = set(similar_users.flatten()) | set(similar_movies)

    # Lọc bỏ các phim đã được người dùng đánh giá
    user_rated_movies = rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index
    final_recommendations = [movie for movie in combined_recommendations if movie not in user_rated_movies]

    return final_recommendations

# Thực hiện gợi ý cho một người dùng và một phim cụ thể
user_id_to_recommend = int(input("Nhập người dùng cần gợi ý phim: "))
movie_id_to_base_recommendation = int(input("Nhập phim cần gợi ý:"))
recommendations = hybrid_recommendation(user_id_to_recommend, movie_id_to_base_recommendation)

# Hiển thị các phim được gợi ý
recommended_movies = movies_df[movies_df['MovieID'].isin(recommendations)][['MovieID', 'Title', 'Genres']]
print(recommended_movies.to_string())




