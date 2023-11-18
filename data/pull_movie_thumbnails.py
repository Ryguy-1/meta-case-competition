import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

OUTPUT_FOLDER = "data/content_thumbnails"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_first_google_image_url(search_phrase) -> str:
    """Returns the URL of the first image result from a Google image search."""
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
    """Downloads an image from a URL and saves it to a file."""
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


if __name__ == "__main__":
    data = pd.read_csv("data/netflix_titles.csv")
    titles = data["title"]
    i = 0
    for title in titles:
        i += 1
        if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{title}.jpg")):
            print(f"Already downloaded image for {title}")
            continue
        image_url = get_first_google_image_url(f"{title} movie poster")
        if image_url:
            save_path = os.path.join(OUTPUT_FOLDER, f"{title}.jpg")
            download_image(image_url, save_path)
        else:
            print(f"No image for title: {title}")
        print(f"{i}/{len(titles)}")
