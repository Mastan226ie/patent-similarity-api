import os
import gdown
import nltk

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            gdown.download(url, filename, quiet=False)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            raise

if __name__ == "__main__":
    file_urls = {
        "embeddings.npy": "https://drive.google.com/uc?id=1FF0Lx3rmop54LJiRvsfUVkPODROEIVXU",
        "patent_index.faiss": "https://drive.google.com/uc?id=1yk-LHEV6-Z8o1zmLKBIxYZns6K9_LuR8",
        "patents.csv": "https://drive.google.com/uc?id=19rOOm_zqC4ZNejKhO1sWMoSu1V_Krce2",
        "preprocessed_patents.csv": "https://drive.google.com/uc?id=1UuI1M0OtaU9JXSE9wBtl3_Ae4jOsHMOB"
    }

    for filename, url in file_urls.items():
        download_file(url, filename)

    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Downloaded NLTK data")
