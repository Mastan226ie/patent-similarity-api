import os
import requests
import nltk

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded {filename}")
        except requests.RequestException as e:
            print(f"Failed to download {filename}: {e}")
            raise

if __name__ == "__main__":
    download_file("https://drive.google.com/uc?export=download&id=1FF0Lx3rmop54LJiRvsfUVkPODROEIVXU", "embeddings.npy")
    download_file("https://drive.google.com/uc?export=download&id=1yk-LHEV6-Z8o1zmLKBIxYZns6K9_LuR8", "patent_index.faiss")
    download_file("https://drive.google.com/uc?export=download&id=19rOOm_zqC4ZNejKhO1sWMoSu1V_Krce2", "patents.csv")
    download_file("https://drive.google.com/uc?export=download&id=1UuI1M0OtaU9JXSE9wBtl3_Ae4jOsHMOB", "preprocessed_patents.csv")
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Downloaded NLTK data")