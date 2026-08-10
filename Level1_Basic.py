"""
Codveda Technologies - Data Analysis Internship
LEVEL 1 (BASIC)
Task 1: Data Cleaning and Preprocessing
Task 2: Exploratory Data Analysis (EDA)

Dataset: house_Prediction_Data_Set.csv (Boston Housing dataset)
Author: Sandra Uzoamaka Ani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------------------------------------
# TASK 1: DATA CLEANING AND PREPROCESSING
# ---------------------------------------------------------

# The raw file has no header row and is whitespace-delimited
# (not comma-delimited, despite the .csv extension) — this is
# exactly the kind of "inconsistent format" the task asks us to handle.
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
           'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv('4__house_Prediction_Data_Set.csv',
                  sep=r'\s+', names=columns)

print("=" * 60)
print("TASK 1: DATA CLEANING")
print("=" * 60)
print(f"\nShape before cleaning: {df.shape}")

# 1. Missing values
missing = df.isnull().sum()
print(f"\nMissing values per column:\n{missing[missing > 0] if missing.sum() > 0 else 'None found'}")

# Even though this dataset is clean, we handle missing values
# programmatically so the pipeline is reusable on messier data:
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())  # median imputation - robust to outliers

# 2. Duplicate rows
dupes = df.duplicated().sum()
print(f"\nDuplicate rows found: {dupes}")
df = df.drop_duplicates()

# 3. Inconsistent formats — CHAS is a binary categorical flag stored as float; cast it properly
df['CHAS'] = df['CHAS'].astype(int)

print(f"\nShape after cleaning: {df.shape}")
print(f"\nData types:\n{df.dtypes}")

# Save cleaned dataset for use in later tasks (Regression, Dashboard)
df.to_csv('/mnt/user-data/outputs/house_data_cleaned.csv', index=False)
print("\nCleaned dataset saved -> house_data_cleaned.csv")

# ---------------------------------------------------------
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Summary statistics
summary = df.describe().T
summary['median'] = df.median(numeric_only=True)
print("\nSummary statistics (mean, std, min/max, median):")
print(summary[['mean', 'median', 'std', 'min', 'max']].round(2))

# Distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(df['MEDV'], kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Distribution of Median Home Value (MEDV, $1000s)')

sns.histplot(df['RM'], kde=True, ax=axes[0, 1], color='seagreen')
axes[0, 1].set_title('Distribution of Average Rooms per Dwelling')

sns.boxplot(y=df['CRIM'], ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('Crime Rate — Boxplot (outlier check)')

sns.scatterplot(x=df['LSTAT'], y=df['MEDV'], ax=axes[1, 1], alpha=0.6)
axes[1, 1].set_title('Lower-Status Population % vs Home Value')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plots/level1_distributions.png', dpi=150)
plt.close()

# Correlation heatmap
plt.figure(figsize=(11, 9))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix — Boston Housing Features')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plots/level1_correlation.png', dpi=150)
plt.close()

# Top correlations with target
target_corr = corr['MEDV'].drop('MEDV').sort_values(key=abs, ascending=False)
print(f"\nFeatures most correlated with home value (MEDV):\n{target_corr.round(3)}")

print("\nPlots saved -> plots/level1_distributions.png, plots/level1_correlation.png")
