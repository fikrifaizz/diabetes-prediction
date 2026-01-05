import kaggle
import os
import zipfile

def run_extract():

    if kaggle.api.authenticate():
        print("Authentication successful")
    
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")

    print("Extracting data...")

    if os.path.exists("data/raw/playground-series-s5e12.zip"):
        os.remove("data/raw/playground-series-s5e12.zip")
    
    kaggle.api.competition_download_files("playground-series-s5e12", path="data/raw")

    with zipfile.ZipFile("data/raw/playground-series-s5e12.zip", "r") as zip_ref:
        zip_ref.extractall("data/raw")

    os.remove("data/raw/playground-series-s5e12.zip")

    print("Extraction completed")

if __name__ == "__main__":
    run_extract()
