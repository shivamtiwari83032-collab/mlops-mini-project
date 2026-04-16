import pandas as pd
import numpy as np 
import yaml
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- LOGGING SETUP ---------------- #
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/feature_engineering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------- LOAD PARAMS ---------------- #
def load_params(path):
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        logging.info("Params loaded successfully")
        return params
    except FileNotFoundError:
        logging.error(f"Params file not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading params: {e}")
        raise

# ---------------- LOAD DATA ---------------- #
def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data {path}: {e}")
        raise

# ---------------- PREPROCESS ---------------- #
def preprocess_data(df):
    try:
        if 'content' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Required columns missing: 'content' or 'sentiment'")

        df.fillna('', inplace=True)
        logging.info("Missing values handled")
        return df

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

# ---------------- FEATURE ENGINEERING ---------------- #
def apply_tfidf(train_df, test_df, max_features):
    try:
        xtrain = train_df['content'].values
        ytrain = train_df['sentiment'].values
        xtest = test_df['content'].values
        ytest = test_df['sentiment'].values

        vectorizer = TfidfVectorizer(max_features=max_features)

        xtrain_bow = vectorizer.fit_transform(xtrain)
        xtest_bow = vectorizer.transform(xtest)

        train_out = pd.DataFrame(xtrain_bow.toarray())
        train_out['label'] = ytrain

        test_out = pd.DataFrame(xtest_bow.toarray())
        test_out['label'] = ytest

        logging.info("TF-IDF transformation completed")
        return train_out, test_out

    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        raise

# ---------------- SAVE DATA ---------------- #
def save_data(df, path):
    try:
        df.to_csv(path, index=False)
        logging.info(f"Saved data to {path}")
    except Exception as e:
        logging.error(f"Error saving file {path}: {e}")
        raise

# ---------------- MAIN PIPELINE ---------------- #
def main():
    try:
        # load params
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']

        # load data
        train_data = load_data('./data/processed/train_processed.csv')
        test_data = load_data('./data/processed/test_processed.csv')

        # preprocess
        train_data = preprocess_data(train_data)
        test_data = preprocess_data(test_data)

        # feature engineering
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # create directory
        data_path = os.path.join('data', 'features')
        os.makedirs(data_path, exist_ok=True)

        # save
        save_data(train_df, os.path.join(data_path, 'train_tfidf.csv'))
        save_data(test_df, os.path.join(data_path, 'test_tfidf.csv'))

        logging.info("Feature engineering pipeline completed successfully")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()