import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from openai_summarizer import predict_movie_category
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm
import torch
import json
import os


def main() -> None:
    # Print torch device
    print(f"Using PyTorch device: {torch.cuda.get_device_name(0)}")

    ############################ LOAD NETFLIX CSV DATA ############################
    netflix_df = pd.read_csv("data/netflix_titles.csv")
    netflix_df["month_year_added"] = pd.to_datetime(
        netflix_df["date_added"], errors="coerce"
    ).dt.to_period("M")
    netflix_titles_to_month_years = (
        netflix_df[["title", "month_year_added"]]
        .dropna()
        .set_index("title")["month_year_added"]
        .astype(str)
        .to_dict()
    )

    ############################ LOAD MOVIE DATA ############################
    with open("data/movie_data.json", "r") as f:
        movie_data = json.load(f)
    overviews = []
    titles = []
    for key, data in movie_data.items():
        if (
            "overview" not in data
            or data["overview"] is None
            or len(data["overview"]) == 0
        ):
            continue
        overviews.append(data["overview"])
        titles.append(key)

    ############################ CLUSTER ############################
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(overviews, show_progress_bar=True)

    # Clustering
    num_clusters = 50  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Dimensionality Reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    ############################ OPENAI PREDICTION ############################
    title_to_cluster = {}
    for title, cluster, overview in zip(titles, cluster_labels, overviews):
        title_to_cluster[title] = {"cluster": str(cluster), "overview": overview}

    # Get cluster category
    cluster_to_category = {}
    cluster_category_file_path = f"generated/cluster_to_category_{num_clusters}.json"
    if not os.path.exists(cluster_category_file_path):
        for cluster in tqdm(range(num_clusters), desc="Predicting cluster categories"):
            cluster_overviews = [
                title_to_cluster[title]["overview"]
                for title in title_to_cluster
                if title_to_cluster[title]["cluster"] == str(cluster)
            ][
                :30
            ]  # only first 30 for openai
            cluster_category = predict_movie_category(cluster_overviews)
            cluster_to_category[str(cluster)] = cluster_category
            print(f"Cluster Number: {cluster}, Category: {cluster_category}")

        # Save cluster_to_category to a JSON file
        with open(cluster_category_file_path, "w") as f:
            json.dump(cluster_to_category, f, indent=4)
    else:
        # Load cluster_to_category from the existing JSON file
        with open(cluster_category_file_path, "r") as f:
            cluster_to_category = json.load(f)

    print(cluster_to_category)

    ############################ PREPARE DATA FOR PLOTTING ############################
    # Data for visualization
    data = {
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "categories": [cluster_to_category[str(label)] for label in cluster_labels],
        "movie_titles": titles,
        "month_year_added": [
            netflix_titles_to_month_years.get(title) for title in titles
        ],  # Map titles to month-year
    }

    # Add Single Dummy Data in Each Category for First Month in Center of Respective Cluster
    dummy_month_year = "2016-01"
    for cluster in range(num_clusters):
        category = cluster_to_category[str(cluster)]
        dummy_title = f"Cluster {cluster} Center"
        data["x"] = np.append(data["x"], 0)
        data["y"] = np.append(data["y"], 0)
        data["categories"] = np.append(data["categories"], category)
        data["movie_titles"] = np.append(data["movie_titles"], dummy_title)
        data["month_year_added"] = np.append(data["month_year_added"], dummy_month_year)

    vis_df = pd.DataFrame(data)

    # Convert 'month_year_added' to Period object for filtering and sorting
    vis_df["month_year_added"] = pd.to_datetime(
        vis_df["month_year_added"]
    ).dt.to_period("M")

    # Filter data from a certain period onwards (e.g., from Jan 2016)
    start_period = pd.Period("2016-01", freq="M")
    vis_df = vis_df[vis_df["month_year_added"] >= start_period]

    # Sort by month-year
    vis_df.sort_values(by="month_year_added", inplace=True)

    # Make the dataset cumulative
    cumulative_dfs = []
    current_period = start_period
    end_period = vis_df["month_year_added"].max()

    while current_period <= end_period:
        period_df = vis_df[vis_df["month_year_added"] <= current_period].copy()
        period_df[
            "month_year_added"
        ] = current_period  # Set all to the current period for animation frame
        cumulative_dfs.append(period_df)
        current_period = current_period + 1  # Increment the period by one month

    cumulative_vis_df = pd.concat(cumulative_dfs)
    cumulative_vis_df["month_year_added"] = cumulative_vis_df[
        "month_year_added"
    ].astype(str)

    ############################ CREATE ANIMATED PLOT ############################
    # Animated Visualization for individual points
    fig = px.scatter(
        cumulative_vis_df,
        x="x",
        y="y",
        animation_frame="month_year_added",  # Updated to reflect month-year
        color="categories",
        hover_name="movie_titles",
        hover_data=["categories"],
        title="Evolution of Netflix Global Catalog",
        labels={"month_year_added": "Date Added to Netflix"},  # Updated label
        # size_max=100,  # Adjust the maximum size of bubbles
    )

    fig.update_xaxes(title_text="Similarity Metric 1")
    fig.update_yaxes(title_text="Similarity Metric 2")
    fig.update_layout(showlegend=True)
    fig.show()

    # Optional: Save the animation as HTML or capture it as a video/gif
    fig.write_html(
        "generated/cluster_individual_month_year_global.html"
    )  # Updated file name


if __name__ == "__main__":
    main()
