import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import re


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
