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
    netflix_df["month_year_added"] = pd.to_datetime(
        netflix_df["date_added"], errors="coerce", infer_datetime_format=True
    ).dt.to_period("M")
    netflix_titles_to_month_year = (
        netflix_df[["title", "month_year_added"]]
        .dropna()
        .set_index("title")["month_year_added"]
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
    num_clusters = 3  # Adjust as needed
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
        "month_year_added": [
            str(netflix_titles_to_month_year.get(title)) for title in titles
        ],
    }
    vis_df = pd.DataFrame(data)

    # Filter out invalid month_year_added values and convert to Period
    vis_df = vis_df.dropna(subset=["month_year_added"])
    vis_df = vis_df[vis_df["month_year_added"] != "NaT"]
    vis_df["month_year_added"] = pd.to_datetime(
        vis_df["month_year_added"], errors="coerce"
    ).dt.to_period("M")
    vis_df = vis_df.dropna(subset=["month_year_added"])
    vis_df = vis_df[
        vis_df["month_year_added"] >= pd.Period("2016-01")
    ]  # Filter data from January 2016 onwards
    vis_df.sort_values(by="month_year_added", inplace=True)  # Sort by month and year

    # Make the dataset cumulative
    cumulative_dfs = []
    all_months = pd.period_range(
        start="2016-01", end=vis_df["month_year_added"].max(), freq="M"
    )
    for month in all_months:
        monthly_df = vis_df[vis_df["month_year_added"] <= month].copy()
        monthly_df["month_year_added"] = str(
            month
        )  # Set all to the current month for animation frame
        cumulative_dfs.append(monthly_df)
    cumulative_vis_df = pd.concat(cumulative_dfs)

    ############################ CREATE ANIMATED PLOT ############################
    # Animated Visualization for individual points
    fig = px.scatter(
        cumulative_vis_df,
        x="x",
        y="y",
        animation_frame="month_year_added",
        color="categories",
        hover_name="movie_titles",
        hover_data=["categories"],
        title="Evolution of Netflix Global Catalog",
        labels={"month_year_added": "Month-Year Added to Netflix"},
        size_max=100,  # Adjust the maximum size of bubbles
    )

    fig.update_layout(showlegend=True)
    fig.show()

    # Optional: Save the animation as HTML or capture it as a video/gif
    fig.write_html("generated/cluster_individual_month_global.html")


if __name__ == "__main__":
    main()
