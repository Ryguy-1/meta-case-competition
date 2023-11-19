import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    os.makedirs("generated", exist_ok=True)
    actor_to_modularitiy_and_color = pd.read_csv(
        "data/gephi_10_modularities_colors.csv"
    )
    netflix_data = pd.read_csv("data/netflix_titles.csv")

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


if __name__ == "__main__":
    main()
