import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="📚 Book Clustering", layout="centered")
st.title("📚 Book Clustering Explorer")
st.markdown("Explore how books group together based on their attributes using KMeans clustering.")


n_clusters = st.slider("Choose number of clusters (K)", min_value=2, max_value=10, value=3, step=1)


np.random.seed(42)
n_books = 180

def generate_books(center, spread, count):
    return np.random.normal(loc=center, scale=spread, size=(count, len(center)))

books_1 = generate_books([120, 4.3, 0.2, 0.1, 2012], [20, 0.2, 0.05, 0.1, 2], n_books // 3)
books_2 = generate_books([350, 4.0, 0.8, 0.6, 2005], [40, 0.3, 0.1, 0.2, 5], n_books // 3)
books_3 = generate_books([420, 3.7, 0.5, 0.9, 2018], [50, 0.25, 0.1, 0.1, 3], n_books // 3)

data = np.vstack([books_1, books_2, books_3])
columns = ['pages', 'avg_rating', 'complexity', 'genre_vector', 'year']
df = pd.DataFrame(data, columns=columns)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)


pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]


fig = px.scatter(
    df,
    x="PCA1",
    y="PCA2",
    color="cluster",
    hover_data=columns,
    title="📊 Book Clusters (Hover for details)",
    color_continuous_scale='Viridis' if n_clusters > 3 else None,
)

st.plotly_chart(fig, use_container_width=True)


with st.expander("🔍 Show raw data"):
    st.dataframe(df)
