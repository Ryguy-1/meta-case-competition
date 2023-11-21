import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from llm_utils import OllamaModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm
import torch
import json
import time


def main() -> None:
    # Print torch device
    print(f"Using PyTorch device: {torch.cuda.get_device_name(0)}")

    ############################ LOAD NETFLIX CSV DATA ############################
    netflix_df = pd.read_csv("data/netflix_titles.csv")
    netflix_df["year_added"] = pd.to_datetime(
        netflix_df["date_added"], errors="coerce", infer_datetime_format=True
    ).dt.year
    netflix_titles_to_years = (
        netflix_df[["title", "year_added"]]
        .dropna()
        .set_index("title")["year_added"]
        .to_dict()
    )

    ############################ LOAD MOVIE DATA ############################
    with open("data/movie_data.json", "r") as f:
        movie_data = json.load(f)
    overviews = []
    titles = []
    for key, data in movie_data.items():
        if "overview" not in data:
            continue
        overviews.append(data["overview"])
        titles.append(key)

    ############################ CLUSTER ############################
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(overviews, show_progress_bar=True)

    # Clustering
    num_clusters = 25  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Dimensionality Reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    ############################ OPENAI PREDICTION ############################
    title_to_cluster = {}
    for title, cluster, overview in zip(titles, cluster_labels, overviews):
        title_to_cluster[title] = {"cluster": str(cluster), "overview": overview}

    ollama_model = OllamaModel(ollama_model_name="mistral-openorca", temperature=0)

    # Predict cluster category
    cluster_to_category = {}
    for cluster in tqdm(range(num_clusters), desc="Predicting cluster categories"):
        cluster_overviews = [
            title_to_cluster[title]["overview"]
            for title in title_to_cluster
            if title_to_cluster[title]["cluster"] == str(cluster)
        ]
        cluster_category = ollama_model.run(cluster_overviews)
        cluster_to_category[str(cluster)] = cluster_category
        print(f"Cluster Number: {cluster}, Category: {cluster_category}")

    ############################ PREPARE DATA FOR PLOTTING ############################
    # Data for visualization
    data = {
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "categories": [cluster_to_category[str(label)] for label in cluster_labels],
        "movie_titles": titles,
        "year_added": [
            netflix_titles_to_years.get(title) for title in titles
        ],  # Map titles to years
    }
    vis_df = pd.DataFrame(data)
    vis_df = vis_df[vis_df["year_added"] >= 2016]  # Filter data from 2016 onwards

    # Group by category and year_added, then calculate mean coordinates and count
    grouped_df = (
        vis_df.groupby(["categories", "year_added"])
        .agg({"x": "mean", "y": "mean", "movie_titles": "count"})
        .reset_index()
    )

    # Making the counts cumulative
    grouped_df["cumulative_count"] = grouped_df.groupby(["categories"])[
        "movie_titles"
    ].cumsum()

    ############################ CREATE ANIMATED PLOT ############################
    # Animated Visualization
    fig = px.scatter(
        grouped_df,
        x="x",
        y="y",
        animation_frame="year_added",
        size="cumulative_count",  # Use cumulative count for bubble size
        color="categories",
        text="categories",  # Display category text
        title="Evolution of Netflix Global Catalog",
        labels={
            "year_added": "Year Added to Netflix",
            "cumulative_count": "Cumulative Number of Movies",
        },
        size_max=100,  # Adjust the maximum size of bubbles
    )

    # Update layout to display category labels in the center of bubbles
    fig.update_traces(textposition="middle center")

    fig.update_layout(showlegend=True)
    fig.show()

    # Optional: Save the animation as HTML or capture it as a video/gif
    fig.write_html("generated/animated_netflix_catalog_evolution.html")

    ############################ YEARLY CHANGE PER CATEGORY PLOT ############################
    # Calculate yearly additions for each category
    yearly_additions_df = (
        vis_df.groupby(["categories", "year_added"]).size().reset_index(name="counts")
    )

    # Create a line plot for yearly additions
    fig_yearly = px.line(
        yearly_additions_df,
        x="year_added",
        y="counts",
        color="categories",
        title="Yearly Additions to Netflix Catalog by Category",
        labels={"year_added": "Year", "counts": "Number of Movies Added"},
    )

    fig_yearly.update_layout(showlegend=True)
    fig_yearly.show()

    # Optional: Save the plot as an HTML file
    fig_yearly.write_html("generated/yearly_netflix_catalog_additions_by_category.html")


if __name__ == "__main__":
    main()
