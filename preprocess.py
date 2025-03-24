import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK resources (only needs to run once, but included for safety)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Please run 'nltk.download()' manually and try again.")
    exit(1)

# Load your dataset (replace 'patents.csv' with your actual file name)
try:
    df = pd.read_csv('patents.csv')
except FileNotFoundError:
    print("Error: 'patents.csv' not found. Please check the file name and path.")
    exit(1)

# Check the first few rows to confirm it loaded correctly
print("Original Data:")
print(df.head())

# Combine Title, Abstract, and Claims into full_text
df['full_text'] = df[['Title', 'Abstract', 'Claims']].fillna('').agg(' '.join, axis=1)

# Define a function to clean text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words('english'))  # Stop words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(tokens)

# Apply cleaning
df['full_text'] = df['full_text'].apply(clean_text)

# Check the cleaned data
print("\nPreprocessed Data:")
print(df[['Patent ID', 'full_text']].head())

# Save preprocessed data
df.to_csv('preprocessed_patents.csv', index=False)
print("\nPreprocessed data saved to 'preprocessed_patents.csv'")

# Optional: Show some stats
print(f"Total patents: {len(df)}")
print(f"Sample full_text: {df['full_text'].iloc[0]}")