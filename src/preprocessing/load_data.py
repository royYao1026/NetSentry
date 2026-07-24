from pathlib import Path
import pandas as pd


def load_ciciot_data(data_path):
    """
    Load CICIoT2023 merged CSV files.

    Parameters:
        data_path (str): path to folder containing Merged*.csv files

    Returns:
        pandas.DataFrame
    """

    data_path = Path(data_path)

    csv_files = sorted(data_path.glob("Merged*.csv"))

    print(f"Found {len(csv_files)} files")

    dataframes = []

    for file in csv_files:
        print(f"Loading {file.name}")

        df = pd.read_csv(file)
        dataframes.append(df)

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    print("Dataset loaded")
    print(data.shape)

    return data


if __name__ == "__main__":

    DATA_PATH = "data/raw/CICIoT2023"

    df = load_ciciot_data(DATA_PATH)

    print(df.head())