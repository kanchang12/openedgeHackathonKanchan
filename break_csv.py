"""
Simple script to read Online Retail.xlsx and split into 3 CSV files
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# READ THE EXCEL FILE
print("Reading Online Retail.xlsx...")
df = pd.read_excel('Online Retail.xlsx')
print(f"Total rows: {len(df)}")

# SPLIT INTO 3 PARTS
# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)

# Second split: 20% test, 10% unknown (from the 30%)
test_df, unknown_df = train_test_split(temp_df, test_size=0.333, random_state=42)

# SAVE AS CSV FILES
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
unknown_df.to_csv('unknown_data.csv', index=False)

# PRINT SIZES
print(f"\nSplit complete:")
print(f"train_data.csv: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"test_data.csv: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
print(f"unknown_data.csv: {len(unknown_df)} rows ({len(unknown_df)/len(df)*100:.1f}%)")
print("\nFiles saved!")