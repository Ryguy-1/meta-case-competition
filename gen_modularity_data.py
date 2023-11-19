import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math


def main():
    os.makedirs("generated", exist_ok=True)
    actor_to_modularitiy_and_color = pd.read_csv(
        "data/gephi_10_modularities_colors.csv"
    )
    netflix_data = pd.read_csv("data/netflix_titles.csv")
    gephi_edges = pd.read_csv("data/col_4_gephi_cast_edges.csv")

    ##### Modularity to Color #####
    modularity_to_color = {
        row["modularity_class"]: row["Color"]
        for _, row in actor_to_modularitiy_and_color.iterrows()
    }

    ##### Actor to Modularity #####
    actor_to_modularity = {  # ACTOR TO MODULARITY
        row["Id"]: row["modularity_class"]
        for _, row in actor_to_modularitiy_and_color.iterrows()
    }

    ##### Modularity to Actors #####
    modularity_to_actors = {}
    for actor, modularity in actor_to_modularity.items():
        if modularity not in modularity_to_actors:
            modularity_to_actors[modularity] = []
        modularity_to_actors[modularity].append(actor)

    ##### Movie to Actors #####
    movie_to_actors = {}
    for movie, actors in zip(netflix_data["title"], netflix_data["cast"]):
        if isinstance(actors, str):
            movie_to_actors[movie] = [x.strip() for x in actors.split(",")]
            continue
        movie_to_actors[movie] = []

    ##### Movie to Production Countries #####
    movie_to_production_countries = {}
    for movie, country in zip(netflix_data["title"], netflix_data["country"]):
        if isinstance(country, str):
            movie_to_production_countries[movie] = [
                x.strip() for x in country.split(",")
            ]
            continue
        movie_to_production_countries[movie] = []

    ##### Actor to Movies #####
    actors_to_movies = {}
    for movie, actors in movie_to_actors.items():
        for actor in actors:
            if actor not in actors_to_movies:
                actors_to_movies[actor] = []
            actors_to_movies[actor].append(movie)

    ############################################
    gen_modularity_color_country_data(
        modularity_to_color,
        actor_to_modularity,
        movie_to_actors,
        movie_to_production_countries,
        "generated/color_to_count_per_country.json",
        "generated/most_common_country_for_each_color.png",
    )

    gen_modularity_num_country_data(
        actor_to_modularity,
        movie_to_actors,
        movie_to_production_countries,
        "generated/modularity_to_count_per_country.json",
    )

    ############################################
    # Calculate isolation metric for each modularity class
    modularity_to_isolation_metric = calculate_isolation_metric(
        actor_to_modularitiy_and_color, gephi_edges
    )
    modularity_to_isolation_metric = {
        modularity: isolation_metric
        for modularity, isolation_metric in modularity_to_isolation_metric.items()
        if not math.isnan(isolation_metric)  # filter out nan values
    }
    # Normalize isolation metric
    max_isolation_metric = max(modularity_to_isolation_metric.values())
    min_isolation_metric = min(modularity_to_isolation_metric.values())
    modularity_to_isolation_metric = {
        modularity: (isolation_metric - min_isolation_metric)
        / (max_isolation_metric - min_isolation_metric)
        for modularity, isolation_metric in modularity_to_isolation_metric.items()
    }
    gen_color_to_isolation_metric_data(
        modularity_to_color,
        modularity_to_isolation_metric,
        "generated/isolation_metric_for_each_color.png",
    )


def gen_modularity_color_country_data(
    modularity_to_color,
    actor_to_modularity,
    movie_to_actors,
    movie_to_production_countries,
    color_to_countries_json_filepath,
    color_to_country_plot_filepath,
):
    """Generate data for modularity to color to country."""
    color_to_count_per_country = {}

    for movie, actors in movie_to_actors.items():
        # Get the set of colors for the movie
        movie_colors = set(
            modularity_to_color[actor_to_modularity[actor]]
            for actor in actors
            if actor in actor_to_modularity
        )

        # Get the list of production countries for the movie
        countries = movie_to_production_countries.get(movie, [])

        # Update counts for each color and country
        for color in movie_colors:
            if color not in color_to_count_per_country:
                color_to_count_per_country[color] = {}
            for country in countries:
                color_to_count_per_country[color][country] = (
                    color_to_count_per_country[color].get(country, 0) + 1
                )

    # 2) Optionally, save the color to count per country map to a file
    with open(color_to_countries_json_filepath, "w") as file:
        json.dump(color_to_count_per_country, file, indent=4)

    # 3) Print top country for each color
    for color, country_to_count in color_to_count_per_country.items():
        print(color, max(country_to_count, key=country_to_count.get))

    # Make visualization of color to most common country (showing the actual colors on the graph is NECESSARY)
    # Extracting the most common country for each color
    color_to_most_common_country = {}
    for color, country_to_count in color_to_count_per_country.items():
        most_common_country = max(country_to_count, key=country_to_count.get)
        color_to_most_common_country[color] = most_common_country

    # Data for plotting
    colors = list(color_to_most_common_country.keys())
    countries = list(color_to_most_common_country.values())

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x=colors, y=[1] * len(colors), palette=colors)
    plt.xticks(range(len(countries)), countries, rotation=45)
    plt.xlabel("Most Common Country")
    plt.ylabel("Count (for visualization purposes)")
    plt.title("Most Common Country for Each Color")
    plt.tight_layout()
    plt.savefig(color_to_country_plot_filepath)
    plt.close()


def gen_modularity_num_country_data(
    actor_to_modularity,
    movie_to_actors,
    movie_to_production_countries,
    modularity_to_num_countries_json_filepath,
):
    """Generate data for modularity raw value to number of countries."""
    modularity_to_count_per_country = {}

    for movie, actors in movie_to_actors.items():
        # Get the set of modularities for the movie
        movie_modularities = set(
            actor_to_modularity[actor]
            for actor in actors
            if actor in actor_to_modularity
        )

        # Get the list of production countries for the movie
        countries = movie_to_production_countries.get(movie, [])

        # Update counts for each modularity and country
        for modularity in movie_modularities:
            if modularity not in modularity_to_count_per_country:
                modularity_to_count_per_country[modularity] = {}
            for country in countries:
                modularity_to_count_per_country[modularity][country] = (
                    modularity_to_count_per_country[modularity].get(country, 0) + 1
                )

    # Optionally, save the modularity to count per country map to a file
    with open(modularity_to_num_countries_json_filepath, "w") as file:
        json.dump(modularity_to_count_per_country, file, indent=4)

    # Print top country for each modularity
    for modularity, country_to_count in modularity_to_count_per_country.items():
        if len(country_to_count) == 0:
            continue
        print(modularity, max(country_to_count, key=country_to_count.get))


def calculate_isolation_metric(modularities_df, edges_df):
    """
    Calculate the isolation metric for each modularity class.

    Args:
    modularities_df (pd.DataFrame): DataFrame containing node IDs and their corresponding modularity classes.
    edges_df (pd.DataFrame): DataFrame containing the edges between nodes.

    Returns:
    dict: A dictionary where keys are modularity classes and values are their isolation metrics.
    """

    # Mapping each node to its modularity class
    node_to_class = dict(
        zip(modularities_df["Id"], modularities_df["modularity_class"])
    )

    # Initializing counters for internal and external edges for each class
    internal_edges = {
        mod_class: 0 for mod_class in modularities_df["modularity_class"].unique()
    }
    external_edges = {
        mod_class: 0 for mod_class in modularities_df["modularity_class"].unique()
    }

    # Counting internal and external edges
    for _, edge in edges_df.iterrows():
        source_class = node_to_class.get(edge["Source"])
        target_class = node_to_class.get(edge["Target"])

        if source_class is not None and target_class is not None:
            if source_class == target_class:
                internal_edges[source_class] += 1
            else:
                external_edges[source_class] += 1
                external_edges[target_class] += 1

    # Calculating isolation metric for each modularity class
    isolation_metric = {}
    for mod_class in internal_edges:
        total_edges = internal_edges[mod_class] + external_edges[mod_class]
        if total_edges > 0:
            isolation_metric[mod_class] = internal_edges[mod_class] / total_edges
        else:
            isolation_metric[mod_class] = float(
                "nan"
            )  # No edges connected to this class

    return dict(
        sorted(
            isolation_metric.items(),
            key=lambda item: (math.isnan(item[1]), item[1]),
            reverse=True,
        )
    )


def gen_color_to_isolation_metric_data(
    modularity_to_color,
    modularity_to_isolation_metric,
    color_to_isolation_metric_plot_filepath,
):
    """
    Generate data and plot for color to isolation metric.
    Args:
    modularity_to_color (dict): Mapping of modularity class to color.
    modularity_to_isolation_metric (dict): Mapping of modularity class to isolation metric.
    color_to_isolation_metric_plot_filepath (str): File path to save the plot.
    """
    # Map each color to its corresponding isolation metric
    color_to_isolation_metric = {}
    for modularity, color in modularity_to_color.items():
        if modularity in modularity_to_isolation_metric:
            color_to_isolation_metric[color] = modularity_to_isolation_metric[
                modularity
            ]

    # Plotting
    colors = list(color_to_isolation_metric.keys())
    isolation_metrics = list(color_to_isolation_metric.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=colors, y=isolation_metrics, palette=colors)
    plt.xlabel("Color")
    plt.ylabel("Isolation Metric")
    plt.title("Isolation Metric for Each Color")
    plt.tight_layout()
    plt.savefig(color_to_isolation_metric_plot_filepath)
    plt.close()


if __name__ == "__main__":
    main()
