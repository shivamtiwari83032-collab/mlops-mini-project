import pandas as pd
import numpy as np
import os
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ---------------- LOGGING SETUP ---------------- #
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/model_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------- LOAD MODEL ---------------- #
def load_model(path):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

# ---------------- LOAD DATA ---------------- #
def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Data loaded from {path}")
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# ---------------- PREPROCESS ---------------- #
def split_data(df):
    try:
        if 'label' not in df.columns:
            raise ValueError("Column 'label' not found in dataset")

        X = df.drop('label', axis=1).values
        y = df['label'].values

        logging.info("Data split into features and labels")
        return X, y

    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

# ---------------- EVALUATE ---------------- #
def evaluate_model(model, X, y):
    try:
        y_pred = model.predict(X)

        # handle models without predict_proba safely
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
        else:
            auc = None
            logging.warning("Model does not support predict_proba, skipping AUC")

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logging.info("Model evaluation completed")
        return metrics

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

# ---------------- SAVE METRICS ---------------- #
def save_metrics(metrics, path):
    try:
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved at {path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

# ---------------- MAIN PIPELINE ---------------- #
def main():
    try:
        # load model
        model = load_model('models/model.pkl')

        # load data
        test_data = load_data('./data/features/test_tfidf.csv')

        # split
        X_test, y_test = split_data(test_data)

        # evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # save
        save_metrics(metrics, 'reports/metrics.json')

        logging.info("Evaluation pipeline completed successfully")

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()