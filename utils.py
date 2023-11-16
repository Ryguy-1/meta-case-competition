import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import re
import yaml
from typing import Dict


def read_yaml_to_dict(yaml_path):
    """
    Reads a YAML file to a dictionary.

    Args:
        yaml_path (str): The path to the YAML file.

    Returns:
        dict: The dictionary representation of the YAML file.
    """
    with open(yaml_path, "r") as file:
        yaml_dict = yaml.safe_load(file)
    return yaml_dict


config = read_yaml_to_dict("config.yml")


def get_first_google_image_url(search_phrase) -> str:
    """
    Returns the URL of the first image result from a Google image search for the given search phrase.

    Args:
        search_phrase (str): The phrase to search for.

    Returns:
        str: The URL of the first image result, or "No image found" if no image was found.
    """
    query = search_phrase.replace(" ", "+")
    url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, "html.parser")
    images = soup.find_all("img", {"src": re.compile("gstatic.com")})
    if images:
        first_image_url = images[0]["src"]
        return first_image_url
    else:
        return "No image found"


def download_image(image_url, save_path) -> bool:
    """
    Downloads an image from the given URL and saves it to the given path.

    Args:
        image_url (str): The URL of the image to download.
        save_path (str): The path to save the image to.

    Returns:
        bool: True if the image was downloaded successfully, False otherwise.
    """
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            return True
        else:
            print(f"Failed to download image. HTTP status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def save_plot(figure, filename) -> None:
    """
    Saves a matplotlib figure to the given filename.

    Args:
        figure (matplotlib.figure.Figure): The figure to save.
        filename (str): The filename to save the figure to.
    """
    figure.savefig(filename, dpi=300)
    plt.close()


def get_actor_details(actor_name) -> Dict:
    """
    Returns the popularity of the given actor from The Movie Database.

    Args:
        actor_name (str): The name of the actor.

    Returns:
        dict: The actor's details from The Movie Database.
    """
    api_key = config["tmdb_api_key"]
    search_url = f"https://api.themoviedb.org/3/search/person?api_key={api_key}&query={actor_name}"
    response = requests.get(search_url).json()
    if response["results"]:
        actor_id = response["results"][0]["id"]
        details_url = (
            f"https://api.themoviedb.org/3/person/{actor_id}?api_key={api_key}"
        )
        details_response = requests.get(details_url).json()
        return details_response
    else:
        return 0
