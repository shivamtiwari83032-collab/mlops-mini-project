import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.model_selection import train_test_split

import logging
os.makedirs('logs', exist_ok=True)
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_params():
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)

        test_size = params['data_ingestion']['test_size']
        logger.debug(f"✅ Loaded test_size from params.yaml: {test_size}")
        return test_size

    except FileNotFoundError:
        logger.error("❌ params.yaml file not found")
        return 0.2  # default fallback

    except KeyError as e:
        logger.error(f"❌ Missing key in params.yaml: {e}")
        return 0.2

    except yaml.YAMLError as e:
        logger.error(f"❌ YAML parsing error: {e}")
        return 0.2

    except Exception as e:
        logger.error(f"❌ Unexpected error in load_params: {e}")
        return 0.2


def read_data(url):
    try:
        df = pd.read_csv(url)
        return df

    except pd.errors.EmptyDataError:
        print("❌ CSV file is empty")
    except pd.errors.ParserError:
        print("❌ Error parsing CSV file")
    except Exception as e:
        print(f"❌ Error reading data: {e}")

    return None


def process_data(df, test_size):
    try:
        if df is None:
            raise ValueError("Input DataFrame is None")

        # Drop column safely
        if 'tweet_id' in df.columns:
            df = df.drop(['tweet_id'], axis=1)

        # Filter required sentiments
        final_df = df[df['sentiment'].isin(['sadness', 'happiness'])].copy()

        # Replace values safely
        final_df['sentiment'] = final_df['sentiment'].replace({
            'sadness': 0,
            'happiness': 1
        })

        # Train-test split
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        return train_data, test_data

    except KeyError as e:
        print(f"❌ Missing column: {e}")
    except ValueError as e:
        print(f"❌ Value error: {e}")
    except Exception as e:
        print(f"❌ Error in processing data: {e}")

    return None, None


def save_data(train_data, test_data):
    try:
        if train_data is None or test_data is None:
            raise ValueError("Train/Test data is None")

        data_path = os.path.join('data', 'raw')

        # Create folder safely
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)

        print("✅ Data saved successfully")

    except PermissionError:
        print("❌ Permission denied while saving files")
    except Exception as e:
        print(f"❌ Error saving data: {e}")


def main():
    try:
        url = "https://github.com/campusx-official/jupyter-masterclass/blob/ab453428bb6cd9c4bcfb8512bcccc36b343f0f52/tweet_emotions.csv?raw=true"

        test_size = load_params()
        df = read_data(url)
        train_data, test_data = process_data(df, test_size)
        save_data(train_data, test_data)

    except Exception as e:
        print(f"❌ Fatal error in main: {e}")


if __name__ == "__main__":
    main()
