import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

df = pd.read_csv('diabetes.csv', na_values=['NA', '?', ''])

print("="*70)
print("1. FIRST FIVE RECORDS (head)")
print("="*70)
print(df.head())

print("\n" + "="*70)
print("2. LAST FIVE RECORDS (tail)")
print("="*70)
print(df.tail())

print("\n" + "="*70)
print("3. SHAPE - TOTAL ROWS AND COLUMNS")
print("="*70)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n" + "="*70)
print("4. COLUMN NAMES")
print("="*70)
print(df.columns.tolist())

print("\n" + "="*70)
print("5. DATA TYPES OF EACH COLUMN")
print("="*70)
print(df.dtypes)

print("\n" + "="*70)
print("6. INFO() - METADATA")
print("="*70)
df.info()

print("\n" + "="*70)
print("7. DESCRIBE() - DESCRIPTIVE STATISTICS")
print("="*70)
print(df.describe())

print("\n" + "="*70)
print("8. MISSING VALUES PER COLUMN")
print("="*70)
missing_count = df.isnull().sum()
print(missing_count)

print("\n" + "="*70)
print("9. PERCENTAGE OF MISSING VALUES PER COLUMN")
print("="*70)
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent.round(2))

print("\n" + "="*70)
print("10. COLUMNS WITH HIGHEST MISSING VALUES")
print("="*70)
print(missing_count.sort_values(ascending=False))

print("\n" + "="*70)
print("SAMPLE QUESTIONS")
print("="*70)
cols_with_missing = missing_count[missing_count > 0].index.tolist()
print(f"Which columns contain missing values?\n{cols_with_missing}")
print(f"\nWhat percentage of data is missing overall? {(df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 100):.2f}%")
print("\nShould missing values be removed or imputed?")
print("If missing percentage per column is small (<5%), rows can be dropped.")
print("If missing percentage is moderate to high, imputation (mean/median for numeric,")
print("mode for categorical) is preferred to avoid losing data.")

print("\n\n" + "#"*70)
print("BASIC EDA OPERATIONS (10)")
print("#"*70)

print("\n" + "="*70)
print("EDA 1. Shape of dataset")
print("="*70)
print(df.shape)

print("\n" + "="*70)
print("EDA 2. Unique value counts per column")
print("="*70)
print(df.nunique())

print("\n" + "="*70)
print("EDA 3. Value counts of target column 'Outcome'")
print("="*70)
print(df['Outcome'].value_counts())

print("\n" + "="*70)
print("EDA 4. Correlation matrix")
print("="*70)
print(df.corr(numeric_only=True))

print("\n" + "="*70)
print("EDA 5. Duplicate rows count")
print("="*70)
print(df.duplicated().sum())

print("\n" + "="*70)
print("EDA 6. Skewness of numeric columns")
print("="*70)
print(df.skew(numeric_only=True))

print("\n" + "="*70)
print("EDA 7. Minimum and maximum values per column")
print("="*70)
print(df.agg(['min', 'max']))

print("\n" + "="*70)
print("EDA 8. Mean, median, mode of numeric columns")
print("="*70)
print("Mean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode().iloc[0])

print("\n" + "="*70)
print("EDA 9. Outlier detection using IQR method")
print("="*70)
numeric_df = df.select_dtypes(include=[np.number])
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
print(outliers)

print("\n" + "="*70)
print("EDA 10. Sample of 5 random rows")
print("="*70)
print(df.sample(5))