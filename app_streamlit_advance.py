import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Sistem Rekomendasi Film",
    page_icon=":popcorn:"
)

movie_list = joblib.load("movie_list.joblib")
# aslinya butuh file "similarity.joblib" tetapi karena ukurannya > 176 MB, kita buat on-the-fly 

cv = CountVectorizer(stop_words="english")
count_matrix = cv.fit_transform(movie_list["tags"].fillna(""))
similarity = cosine_similarity(count_matrix)

titles = movie_list["title"].values

def recommend(movie):
    # cari index dari judul yang dipilih
    index = movie_list[movie_list["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, reverse=True, key=lambda x: x[1])

    # ambil 5 film teratas (skip index 0 karena itu filmnya sendiri)
    rekomendasi = [movie_list.iloc[i[0]].title for i in distances[1:6]]

    # tampilkan sebagai list bernomor
    hasil = "\n".join([f"{idx+1}. {judul}" for idx, judul in enumerate(rekomendasi)])
    st.success(f"5 Rekomendasi Teratas:\n\n{hasil}")


st.title(":popcorn: Sistem Rekomendasi Film")
st.markdown("Aplikasi machine learning recommendation system dengan konsep "
            "**Content-based Filtering dengan CountVectorizer dan Cosine Similarity**")

selected_movie = st.selectbox("Film apa yang kamu suka?", titles)

if st.button("Tampilkan Rekomendasi", type="primary"):
    recommend(selected_movie)
    st.balloons()

st.divider()
st.caption("Dibuat dengan :fire: oleh **Adi Setiawan**")
