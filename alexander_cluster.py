import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
import torch
import json


def main() -> None:
    # Print torch device
    print(f"Using PyTorch device: {torch.cuda.get_device_name(0)}")
    # Load data
    with open("data/movie_data.json", "r") as f:
        movie_data = json.load(f)
    overviews = []
    titles = []
    for key, data in movie_data.items():
        if "overview" not in data:
            continue
        overviews.append(data["overview"])
        titles.append(key)

    # Generate embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(overviews, show_progress_bar=True)

    # Clustering
    num_clusters = 20  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Dimensionality Reduction
    tsne = TSNE(n_components=2, random_state=42)  # Use 3 for 3D visualization
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Data for visualization
    data = {
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "cluster_labels": [str(x) for x in cluster_labels],
        "movie_titles": titles,
    }
    vis_df = pd.DataFrame(data)

    # Visualization
    fig = px.scatter(
        vis_df, x="x", y="y", color="cluster_labels", hover_name="movie_titles"
    )
    fig.update_layout(title="t-SNE Visualization with K-Means Clusters")
    fig.show()


if __name__ == "__main__":
    main()
