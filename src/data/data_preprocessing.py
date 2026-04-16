import pandas as pd
import numpy as np
import os 
import re
import nltk  
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- LOGGING SETUP ---------------- #
logging.basicConfig(
    filename='logs/data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# create logs directory if not exists
os.makedirs('logs', exist_ok=True)

# ---------------- DATA LOADING ---------------- #
def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Successfully loaded data from {path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading file {path}: {e}")
        raise

# ---------------- NLTK DOWNLOAD ---------------- #
def download_nltk():
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logging.info("NLTK resources downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

# ---------------- TEXT CLEANING FUNCTIONS ---------------- #
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    except Exception as e:
        logging.warning(f"Lemmatization failed: {e}")
        return text

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in text.split() if word not in stop_words])
    except Exception as e:
        logging.warning(f"Stopword removal failed: {e}")
        return text

def remove_numbers(text):
    try:
        return ''.join([i for i in text if not i.isdigit()])
    except Exception as e:
        logging.warning(f"Number removal failed: {e}")
        return text

def lower_case(text):
    try:
        return ' '.join([word.lower() for word in text.split()])
    except Exception as e:
        logging.warning(f"Lowercase conversion failed: {e}")
        return text

def remove_punctuation(text):
    try:
        return text.translate(str.maketrans('', '', string.punctuation))
    except Exception as e:
        logging.warning(f"Punctuation removal failed: {e}")
        return text

def remove_urls(text):
    try:
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    except Exception as e:
        logging.warning(f"URL removal failed: {e}")
        return text

def remove_small_sentences(text):
    try:
        return ' '.join([word for word in text.split() if len(word) > 2])
    except Exception as e:
        logging.warning(f"Small word removal failed: {e}")
        return text

# ---------------- NORMALIZATION ---------------- #
def normalize_text(df):
    try:
        if 'content' not in df.columns:
            raise ValueError("Column 'content' not found in dataframe")

        df['content'] = df['content'].astype(str)

        df['content'] = df['content'].apply(lemmatization)
        df['content'] = df['content'].apply(remove_stopwords)
        df['content'] = df['content'].apply(remove_numbers)
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_punctuation)
        df['content'] = df['content'].apply(remove_urls)
        df['content'] = df['content'].apply(remove_small_sentences)

        logging.info("Text normalization completed successfully")
        return df

    except Exception as e:
        logging.error(f"Error in text normalization: {e}")
        raise

# ---------------- SAVE DATA ---------------- #
def save_data(df, path):
    try:
        df.to_csv(path, index=False)
        logging.info(f"Data saved successfully at {path}")
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")
        raise

# ---------------- MAIN PIPELINE ---------------- #
def main():
    try:
        # load data
        train_data = load_data('./data/raw/train.csv')
        test_data = load_data('./data/raw/test.csv')

        # download nltk
        download_nltk()

        # process
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        # create directory
        data_path = os.path.join('data', 'processed')
        os.makedirs(data_path, exist_ok=True)

        # save
        save_data(train_processed, os.path.join(data_path, 'train_processed.csv'))
        save_data(test_processed, os.path.join(data_path, 'test_processed.csv'))

        logging.info("Pipeline executed successfully")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()