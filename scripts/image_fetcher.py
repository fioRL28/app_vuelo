# image_fetcher.py
import requests

UNSPLASH_ACCESS_KEY = ""  # Pega aquí tu clave de Unsplash

def get_image_url(destination):
    query = destination.replace(" ", "+")
    url = f"https://api.unsplash.com/search/photos?query={query}&client_id={UNSPLASH_ACCESS_KEY}&orientation=landscape"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            return data["results"][0]["urls"]["regular"]
    return ""
