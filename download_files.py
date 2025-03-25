import requests
import os
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
# Download large files at startup
download_file("https://drive.google.com/uc?export=download&id=1FF0Lx3rmop54LJiRvsfUVkPODROEIVXU", "embeddings.npy")
download_file("https://drive.google.com/uc?export=download&id=1yk-LHEV6-Z8o1zmLKBIxYZns6K9_LuR8", "patent_index.faiss")
download_file("https://drive.google.com/uc?export=download&id=19rOOm_zqC4ZNejKhO1sWMoSu1V_Krce2", "patents.csv")
download_file("https://drive.google.com/uc?export=download&id=1UuI1M0OtaU9JXSE9wBtl3_Ae4jOsHMOB", "preprocessed_patents.csv")