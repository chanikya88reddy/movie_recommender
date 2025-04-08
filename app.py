import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
@st.cache_data
def load_data():
   
    movies =  pd.read_csv("movies.csv")
    credits = pd.read_csv("credits.csv")
    df = movies.merge(credits, left_on="id",right_on="movie_id")
    df = df[["title_y", "overview", "genres", "cast", "crew", "id"]]
    df.columns = ["title", "overview", "genres", "cast", "crew", "id"]
    df.dropna(inplace=True)
    return df
#preproccesing
def preprocess(df):
    df["combined"] = df["overview"]+ " "+ df["genres"]+" "+ df["cast"]+ " "+ df["crew"]
    return df
#feature extracting
def get_vectorizer(data, method = "TF-IDF"):
    if method == "TF-IDF":
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        vectors = vectorizer.fit_transform(data["combined"])
    else:
        vectorizer= HashingVectorizer(stop_words="english", n_features = 5000, alternate_sign= False)
        vectors = vectorizer.transform(data["combined"])
    return vectors

#KMeans Clustering
def cluster_movies(vectors, df, n_clusters = 10):
    model = KMeans(n_clusters=n_clusters,random_state=42)
    df["clusters"] = model.fit_predict(vectors)
    return df, model

#Recommend movies by cosine similarity
def recommend(title, vectors, data, top_n = 5):
    idx = data[data["title"]== title].index[0]
    sim = cosine_similarity(vectors[idx], vectors).flatten()
    scores = list(enumerate(sim))
    scores = sorted(scores, key = lambda x: x[1], reverse = True)[1:top_n+1]
    return data.iloc[[i[0] for i in scores]]

#stremlit UI

st.title("🎬Movie Recommendation system")
df = load_data()
df = preprocess(df)
method = st.selectbox("select ML method", ["KMeans", "Cosine_Similarity"])
vector_method = st.selectbox("Select vector method", ["TF-IDF", "Feature Hashing"])
vectors = get_vectorizer(df, vector_method)

if method == "Clustering":
    n_clusters = st.slider("select niumber of clusters", 2, 20, 10)
    df,model = cluster_movies(vectors, df, n_clusters)
    selected_cluster = st.selectbox("Choose a cluster", list(range(n_clusters)))
    cluster_df = df[df["cluster"] == selected_cluster]
    st.write(cluster_df[["title"]].head(10))
else:
    movie_title = st.selectbox("Pick a movie", df["title"].values)
    top_n = st.slider("Number of recommendations", 1, 10, 5)
    results = recommend(movie_title, vectors, df, top_n)
    
    st.subheader("🔍 Top Recommended Movies")
    for i, row in results.iterrows():
        st.markdown(f"**🎞️ {row['title']}**")
        if st.button(f"⭐ Save '{row['title']}' to Favorites", key=i):
            st.success(f"✅ Added '{row['title']}' to favorites!")


