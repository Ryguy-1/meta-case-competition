import pandas as pd
import json
import os


def main():
    os.makedirs("generated", exist_ok=True)
    actor_to_modularitiy_and_color = pd.read_csv(
        "data/gephi_10_modularities_colors.csv"
    )

    modularity_to_color = {  # MODULARITY TO COLOR
        row["modularity_class"]: row["color"]
        for _, row in actor_to_modularitiy_and_color.iterrows()
    }

    actor_to_modularity = {  # ACTOR TO MODULARITY
        row["Id"]: row["modularity_class"]
        for _, row in actor_to_modularitiy_and_color.iterrows()
    }

    modularity_to_actors = {}  # MODULARITY TO ACTORS
    for actor, modularity in actor_to_modularity.items():
        if modularity not in modularity_to_actors:
            modularity_to_actors[modularity] = []
        modularity_to_actors[modularity].append(actor)

    netflix_data = pd.read_csv("data/netflix_titles.csv")
    movie_to_actors = {}  # MOVIE TO ACTORS
    for movie, actors in zip(netflix_data["title"], netflix_data["cast"]):
        if isinstance(actors, str):
            movie_to_actors[movie] = [x.strip() for x in actors.split(",")]
            continue
        movie_to_actors[movie] = []

    movie_to_description = {  # MOVIE TO DESCRIPTION
        row["title"]: row["description"] for _, row in netflix_data.iterrows()
    }
    movie_to_production_country = {}  # MOVIE TO PRODUCTION COUNTRY
    for movie, country in zip(netflix_data["title"], netflix_data["country"]):
        if isinstance(country, str):
            movie_to_production_country[movie] = country
            continue
        movie_to_production_country[movie] = ""

    actors_to_movies = {}  # ACTOR TO MOVIES
    for movie, actors in movie_to_actors.items():
        for actor in actors:
            if actor not in actors_to_movies:
                actors_to_movies[actor] = []
            actors_to_movies[actor].append(movie)

    ############################################
    # 1) make map of each color to the countries of movies produced in that color (with the number of movies produced in that color per country)
    color_to_count_per_country = {}

    # Iterate over each movie and its production country
    for movie, country in movie_to_production_country.items():
        # Get the modularity class of the movie based on its actors
        modularity_classes = set(
            actor_to_modularity[actor] for actor in movie_to_actors.get(movie, [])
        )

        # For each modularity class, update the count in the color to country map
        for modularity in modularity_classes:
            color = modularity_to_color.get(modularity)

            if color not in color_to_count_per_country:
                color_to_count_per_country[color] = {}

            if country not in color_to_count_per_country[color]:
                color_to_count_per_country[color][country] = 0

            color_to_count_per_country[color][country] += 1

    # 2) Optionally, save the color to count per country map to a file
    with open("generated/color_to_count_per_country.json", "w") as file:
        json.dump(color_to_count_per_country, file, indent=4)


if __name__ == "__main__":
    main()
