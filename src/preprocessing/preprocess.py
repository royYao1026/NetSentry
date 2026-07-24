from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ciciot_sample.csv"
)

LABEL_COLUMN = "Label"
TARGET_COLUMN = "is_attack"
RANDOM_STATE = 42


def load_binary_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and clean the sampled CICIoT2023 dataset.

    Binary target:
        BENIGN -> 0
        every attack label -> 1

    Cleaning:
        1. Convert infinity to missing values.
        2. Remove feature patterns with conflicting binary labels.
        3. Remove duplicate feature/target rows.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Required column '{LABEL_COLUMN}' was not found."
        )

    original_rows = len(df)

    # Normalize label text.
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    feature_columns = [
        column
        for column in df.columns
        if column != LABEL_COLUMN
    ]

    # Convert positive and negative infinity to missing values.
    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Create binary target before duplicate handling.
    df[TARGET_COLUMN] = (
        df[LABEL_COLUMN]
        .ne("BENIGN")
        .astype("int8")
    )

    # Create an identifier for each unique feature pattern.
    df["_feature_hash"] = pd.util.hash_pandas_object(
        df[feature_columns],
        index=False,
    )

    # Find identical feature patterns assigned both binary labels.
    target_counts = (
        df.groupby("_feature_hash")[TARGET_COLUMN]
        .nunique()
    )

    conflicting_hashes = target_counts[
        target_counts > 1
    ].index

    conflicting_mask = df["_feature_hash"].isin(
        conflicting_hashes
    )

    conflicting_pattern_count = len(conflicting_hashes)
    conflicting_row_count = int(conflicting_mask.sum())

    # Remove ambiguous feature patterns.
    df = df.loc[~conflicting_mask].copy()

    rows_before_deduplication = len(df)

    # Remove records with identical features and binary targets.
    df = (
        df.drop_duplicates(
            subset=feature_columns + [TARGET_COLUMN]
        )
        .reset_index(drop=True)
    )

    removed_duplicates = (
        rows_before_deduplication - len(df)
    )

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    non_numeric_columns = list(
        X.select_dtypes(exclude="number").columns
    )

    if non_numeric_columns:
        raise TypeError(
            "Non-numeric feature columns found: "
            f"{non_numeric_columns}"
        )

    print("Dataset cleaning summary")
    print("------------------------")
    print(f"Original rows:                  {original_rows:,}")
    print(
        "Conflicting feature groups:     "
        f"{conflicting_pattern_count:,}"
    )
    print(
        "Rows removed for conflicts:     "
        f"{conflicting_row_count:,}"
    )
    print(
        "Duplicate binary rows removed:  "
        f"{removed_duplicates:,}"
    )
    print(f"Remaining rows:                 {len(df):,}")
    print(f"Number of features:             {X.shape[1]}")
    print(
        "Missing values:                 "
        f"{int(X.isna().sum().sum()):,}"
    )

    return X, y


def create_binary_splits(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Create stratified train, validation and test sets.

    Split:
        training:   70%
        validation: 15%
        test:       15%
    """
    X, y = load_binary_dataset(data_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def print_split_summary(
    name: str,
    target: pd.Series,
) -> None:
    """Print benign and attack counts for one split."""
    counts = target.value_counts().sort_index()

    benign_count = int(counts.get(0, 0))
    attack_count = int(counts.get(1, 0))

    benign_percentage = benign_count / len(target) * 100
    attack_percentage = attack_count / len(target) * 100

    print(
        f"{name:<12} "
        f"rows={len(target):>8,} | "
        f"benign={benign_count:>7,} "
        f"({benign_percentage:>5.2f}%) | "
        f"attack={attack_count:>8,} "
        f"({attack_percentage:>5.2f}%)"
    )


if __name__ == "__main__":
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = create_binary_splits()

    print("\nBinary split summary")
    print("--------------------")
    print_split_summary("Train", y_train)
    print_split_summary("Validation", y_validation)
    print_split_summary("Test", y_test)

    print("\nFeature matrix shapes")
    print("---------------------")
    print(f"X_train:      {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test:       {X_test.shape}")