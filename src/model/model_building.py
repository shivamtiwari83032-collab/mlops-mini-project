import pandas as pd
import numpy as np
import os
import pickle
import yaml
import logging
from sklearn.ensemble import GradientBoostingClassifier

# ---------------- LOGGING SETUP ---------------- #
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------- LOAD PARAMS ---------------- #
def load_params(path):
    try:
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        logging.info("Parameters loaded successfully")
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
        logging.info(f"Data loaded from {path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data {path}: {e}")
        raise

# ---------------- PREPROCESS ---------------- #
def split_features_labels(df):
    try:
        if 'label' not in df.columns:
            raise ValueError("Column 'label' not found in dataset")

        X = df.drop('label', axis=1).values
        y = df['label'].values

        logging.info("Features and labels split successfully")
        return X, y

    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

# ---------------- TRAIN MODEL ---------------- #
def train_model(X, y, n_estimators, learning_rate):
    try:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        model.fit(X, y)

        logging.info("Model trained successfully")
        return model

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

# ---------------- SAVE MODEL ---------------- #
def save_model(model, path):
    try:
        # create directory if not exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

        logging.info(f"Model saved at {path}")

    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise
# ---------------- MAIN PIPELINE ---------------- #
def main():
    try:
        # load params
        params = load_params('params.yaml')
        n_estimators = params['model_building']['n_estimators']
        learning_rate = params['model_building']['learning_rate']

        # load data
        train_data = load_data('./data/features/train_tfidf.csv')

        # split
        X_train, y_train = split_features_labels(train_data)

        # train
        model = train_model(X_train, y_train, n_estimators, learning_rate)

        # create model directory
        os.makedirs('models', exist_ok=True)

        # save
        save_model(model, 'models/model.pkl')


        logging.info("Training pipeline completed successfully")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        print(f"ERROR: {e}")   # show in terminal
        raise                 # VERY IMPORTANT

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()