import pandas as pd
import os
import json
import concurrent.futures
import time
from typing import Dict
import requests
from api_keys import tmdb as config

OUTPUT_FILENAME = "col_4_director_data.json"


def load_current_director_data() -> Dict:
    """Load current director data from file."""
    if not os.path.exists(OUTPUT_FILENAME):
        return {}
    with open(OUTPUT_FILENAME, "r") as file:
        return json.load(file)


def save_current_director_data(stats: Dict):
    """Save current director data to file."""
    with open(OUTPUT_FILENAME, "w") as file:
        json.dump(stats, file, indent=4)


def get_director_details(director_name: str) -> Dict:
    """Get director details from TMDB API."""
    api_key = config["tmdb_api_key"]
    search_url = f"https://api.themoviedb.org/3/search/person?api_key={api_key}&query={director_name}"
    response = requests.get(search_url).json()
    if response["results"]:
        director_id = response["results"][0]["id"]
        details_url = (
            f"https://api.themoviedb.org/3/person/{director_id}?api_key={api_key}"
        )
        details_response = requests.get(details_url).json()
        return details_response
    else:
        return {}


if __name__ == "__main__":
    data = pd.read_csv("netflix_titles.csv")
    all_directors = data["director"].dropna()
    futures = {}
    director_details = {}

    # Load Director Data from File (initialize)
    director_details = load_current_director_data()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for director in all_directors:
            if director in director_details:
                print(f"Skipping {director}. Already got their data.")
                continue
            future = executor.submit(get_director_details, director)
            futures[future] = director
        while futures:
            done, _ = concurrent.futures.wait(
                futures, timeout=3, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                director = futures.pop(future)
                director_details[director] = future.result()
            save_current_director_data(director_details)
            time.sleep(10)
            print(
                f"Got {len(director_details)}/{len(all_directors)} directors' data so far..."
            )
    save_current_director_data(director_details)
