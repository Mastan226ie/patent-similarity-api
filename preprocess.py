import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# Ensure necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Check if patents.csv exists
if not os.path.exists('patents.csv'):
    print("Error: 'patents.csv' not found. Make sure it's uploaded to the deployment environment.")
    exit(1)

# Load dataset
df = pd.read_csv('patents.csv')

# Combine Title, Abstract, and Claims into full_text
df['full_text'] = df[['Title', 'Abstract', 'Claims']].fillna('').agg(' '.join, axis=1)


# Function to clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)

    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Apply cleaning
df['full_text'] = df['full_text'].apply(clean_text)

# Save processed data
df.to_csv('preprocessed_patents.csv', index=False)
print("Preprocessed data saved to 'preprocessed_patents.csv'.")

# Show dataset summary
print(f"Total patents: {len(df)}")
print(f"Sample full_text: {df['full_text'].iloc[0]}")
