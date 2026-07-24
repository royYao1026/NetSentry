from pathlib import Path
import pandas as pd


def create_sample(
    input_path,
    output_path,
    sample_size=500000
):
    """
    Create a smaller dataset sample
    from CICIoT2023 raw data.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    csv_files = sorted(
        input_path.glob("Merged*.csv")
    )

    print(f"Found {len(csv_files)} files")

    samples = []

    rows_per_file = sample_size // len(csv_files)

    for file in csv_files:
        print(f"Processing {file.name}")

        df = pd.read_csv(file)

        sample = df.sample(
            n=min(rows_per_file, len(df)),
            random_state=42
        )

        samples.append(sample)


    result = pd.concat(
        samples,
        ignore_index=True
    )


    result = result.sample(
        frac=1,
        random_state=42
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    result.to_csv(
        output_path,
        index=False
    )


    print("Sample created")
    print(result.shape)


if __name__ == "__main__":

    INPUT = "data/raw/CICIoT2023"

    OUTPUT = (
        "data/processed/"
        "ciciot_sample.csv"
    )

    create_sample(
        INPUT,
        OUTPUT,
        sample_size=500000
    )