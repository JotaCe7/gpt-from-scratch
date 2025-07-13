import os
import urllib.request


def load_data(file_path: str, url: str = None) -> str:
    """
    Downloads a text file if it doesn't exist locally, then reads and return its content.

    Args:
        file_path (str): The local path where the file is stored or will be saved
        url (str): The URL of the text file to download if it doesn't exist locally.

    Returns:
        str: The content of the text file as string.
    """
    if not os.path.exists(file_path):
        print(f"Downloading data from {url}...")
        with urllib.request.urlopen(url) as resposne:
            text_data = resposne.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        print("Download complete.")
    else:
        print(f"File '{file_path}' already exists. Loading fomr disk...")
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        print("Load complete.")
    
    return text_data