import pandas as pd
import os
import json
import concurrent.futures
import time
from typing import Dict
import requests
from api_keys import tmdb as config

OUTPUT_FILENAME = "movie_data.json"


def load_current_movie_data() -> Dict:
    """Load current movie data from file."""
    if not os.path.exists(OUTPUT_FILENAME):
        return {}
    with open(OUTPUT_FILENAME, "r") as file:
        return json.load(file)


def save_current_movie_data(stats: Dict):
    """Save current movie data to file."""
    with open(OUTPUT_FILENAME, "w") as file:
        json.dump(stats, file, indent=4)


def get_movie_details(movie_title: str) -> Dict:
    """Get movie details from TMDB API."""
    api_key = config["tmdb_api_key"]
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    response = requests.get(search_url).json()
    if response["results"]:
        movie_id = response["results"][0]["id"]
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        details_response = requests.get(details_url).json()
        return details_response
    else:
        return {}


if __name__ == "__main__":
    data = pd.read_csv("netflix_titles.csv")
    all_movies = data["title"].dropna()
    futures = {}
    movie_details = {}

    # Load Movie Data from File (initialize)
    movie_details = load_current_movie_data()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for movie in all_movies:
            if movie in movie_details:
                print(f"Skipping {movie}. Already got its data.")
                continue
            future = executor.submit(get_movie_details, movie)
            futures[future] = movie
        while futures:
            done, _ = concurrent.futures.wait(
                futures, timeout=3, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                movie = futures.pop(future)
                movie_details[movie] = future.result()
            save_current_movie_data(movie_details)
            time.sleep(10)
            print(
                f"Got {len(movie_details)}/{len(all_movies)} movies' data so far..."
            )
    save_current_movie_data(movie_details)
