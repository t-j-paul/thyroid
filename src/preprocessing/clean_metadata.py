"""
Clean and integrate structured metadata.
Usage: python clean_metadata.py --metadata_csv data/raw/metadata.csv --output_csv data/processed/clean_metadata.csv
"""

import pandas as pd
import numpy as np
import argparse

def clean_metadata(metadata_csv, output_csv):
    df = pd.read_csv(metadata_csv)
    # Example cleaning steps
    # Drop rows with missing critical fields
    df = df.dropna(subset=['patient_id', 'nodule_label'])
    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Cap/floor age values
    df['age'] = np.clip(df['age'], 0, 120)
    # Convert nodule_label to binary
    df['nodule_label'] = df['nodule_label'].map({'benign': 0, 'malignant': 1})
    df = df[df['nodule_label'].isin([0, 1])]
    # Fill missing lesion_size with median
    if 'lesion_size' in df.columns:
        df['lesion_size'] = df['lesion_size'].fillna(df['lesion_size'].median())
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Cleaned metadata saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()
    clean_metadata(args.metadata_csv, args.output_csv)

if __name__ == '__main__':
    main()
