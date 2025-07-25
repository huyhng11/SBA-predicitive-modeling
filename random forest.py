# -*- coding: utf-8 -*-
"""EDA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1udnPt0mfu7hlfQBEe8B4SclP9eQ7IoVN

1. **Load and Run correlation matrix**
"""


import pandas as pd
import matplotlib.pyplot as plt

# 1) Load
df = pd.read_csv('SBAnational.csv')
df.head()
print(df.dtypes)

# Clean currency columns
currency_cols = ['DisbursementGross', 'BalanceGross', 'GrAppv', 'SBA_Appv', 'ChgOffPrinGr']
for col in currency_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('[\$,]', '', regex=True), errors='coerce')

# Parse datetime columns
df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'], errors='coerce')
df['DisbursementDate'] = pd.to_datetime(df['DisbursementDate'], errors='coerce')
df['ChgOffDate'] = pd.to_datetime(df['ChgOffDate'], errors='coerce')

# Target variable
df['MIS_Status_Binary'] = df['MIS_Status'].map({'P I F': 0, 'CHGOFF': 1})
df['LowDoc_Binary'] = df['LowDoc'].map({'Y': 1, 'N': 0})
df['RevLineCr_Binary'] = df['RevLineCr'].map({'Y': 1, 'N': 0})
df['UrbanRural_Numeric'] = pd.to_numeric(df['UrbanRural'], errors='coerce')

numeric_features = [
    'Term', 'NoEmp', 'CreateJob', 'RetainedJob',
    'FranchiseCode', 'DisbursementGross', 'BalanceGross',
    'ChgOffPrinGr', 'GrAppv', 'SBA_Appv',
    'LowDoc_Binary', 'RevLineCr_Binary', 'UrbanRural_Numeric',
    'MIS_Status_Binary'
]

df_corr = df[numeric_features].dropna()

# Compute correlation matrix
correlation_matrix = df_corr.corr()
print(correlation_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Selected Numeric Features", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import numpy as np

# Mask to get only upper triangle, to avoid duplicate pairs
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

# Extract correlation values from upper triangle
corr_pairs = correlation_matrix.where(mask).stack()

# Filter pairs with correlation > 0.4
strong_corr = corr_pairs[corr_pairs > 0.4]

print("Correlation pairs with correlation > 0.4:")
print(strong_corr)

"""2. **New vs Established Business**"""

df['NewExist'].value_counts()

df_filtered = df[df['NewExist'].isin([1, 2])]

default_rates = df_filtered.groupby('NewExist')['MIS_Status_Binary'].mean()

business_type_map = {1: 'New', 2: 'Established'}
default_rates.index = default_rates.index.map(business_type_map)

print("Default rate by business status:")
print(default_rates)

# Plot default rates
default_rates.plot(kind='bar', color=['orange', 'skyblue'])
plt.ylabel('Default Rate')
plt.title('Default Rate: New vs Established Businesses')
plt.xticks(rotation=0)
plt.show()

"""The default rate of 2 business models does not differ much, suggesting that business longevity would not be a good indicator for default rate

3. **Loans Backed by Real Estate**
"""

df['RealEstate']=df['Term'].apply(lambda x: 1 if x >= 240 else 0)
df['RealEstate'].value_counts()

grouped = df.groupby('RealEstate')['MIS_Status_Binary']
default_rate = grouped.mean()
pif_rate     = 1 - default_rate

rates = pd.DataFrame({
    'Default Rate': default_rate,
    'Pay-in-Full Rate': pif_rate
}).rename(index={0: 'Non Real Estate Backed (<240m)', 1: 'Real Estate Backed (≥240m)'})

print(rates)

rates.plot(kind='bar', figsize=(8,5))
plt.ylim(0,1)
plt.ylabel('Rate')
plt.title('Default vs. Pay-in-Full Rates\nby Real-Estate Backing')
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

"""Loans backed by real estate exhibit a dramatically lower default rate (1.6% vs. 20.8%) and a correspondingly higher pay-in-full rate (98.4% vs. 79.2%). This aligns perfectly with the collateral hypothesis: long-term, real-estate-secured loans carry far less risk of charge-off because the underlying property value typically covers outstanding principal

4. **Economic Recession**
"""

recession_start = pd.to_datetime('2007-12-01')
recession_end   = pd.to_datetime('2009-06-30')

# Make sure ApprovalDate is datetime and drop rows missing it
df = df[df['ApprovalDate'].notna()]

# Create Recession dummy:
df['Recession'] = df['ApprovalDate'].between(recession_start, recession_end).astype(int)
print(df['Recession'].value_counts())

# Compute default and pay-in-full rates by Recession flag
grouped = df.groupby('Recession')['MIS_Status_Binary']
default_rate = grouped.mean()            # mean of 1/0 default flags
pif_rate     = 1 - default_rate

rates = pd.DataFrame({
    'Default Rate': default_rate,
    'Pay-in-Full Rate': pif_rate
}).rename(index={0: 'Outside Recession', 1: 'During Recession'})

print(rates)

ax = rates.plot(
    kind='bar',
    stacked=True,
    figsize=(8, 5)
)

ax.set_ylabel('Proportion')
ax.set_ylim(0, 1)
ax.set_title('Loan Outcomes by Recession Period (Stacked)')
ax.legend(loc='upper right')

for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', label_type='center')

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

"""- The patter is clear, during recession companies have higher default rate and lower PIF. Therefore, it is suggested that loan performance is highly sensitive to macroeconomic conditions. Underwriting standards that look sound in growth periods can yield sharply higher losses in downturns
- Regulators or risk managers should stress‐test portfolios against recession-like scenarios—SBA portfolios could see defaults more than double under severe downturns

5. **SBA’s Guaranteed Portion of Approved Loan**
"""

df_sba = df[['SBA_Appv', 'MIS_Status']].dropna()
df_sba = df[df['MIS_Status'].isin(['P I F', 'CHGOFF'])]

# Visualize using a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_sba, x='MIS_Status', y='SBA_Appv')
plt.title('Distribution of SBA Guaranteed Amount by Loan Status')
plt.xlabel('MIS_Status (P I F = Paid in Full, CHGOFF = Charged Off)')
plt.ylabel('SBA Guaranteed Amount ($)')
plt.grid(True)
plt.show()

"""- The median SBA guaranteed amount is higher for PIF loans than for CHGOFF loans --> Suggests that loans with higher guaranteed amounts are more likely to be paid back in full

- The IQR has the box height for PIF is larger, showing more variability among successfully repaid loans --> PIF have more outliers while Charge Off have less, indicating less success rate when the SBA's loan proportion covered is less

6. **Default Rate by Industry, State**
"""

state_stats = df.groupby('State').agg(
    loan_count      = ('MIS_Status_Binary','size'),
    default_rate    = ('MIS_Status_Binary','mean')
).reset_index()

top10 = state_stats.sort_values('default_rate', ascending=False).head(10)

print("Top 10 States by SBA Default Rate :")
print(top10[['State','loan_count','default_rate']])

plt.bar(top10['State'], top10['default_rate'])
plt.ylim(0, top10['default_rate'].max() * 1.1)
plt.ylabel('Default Rate')
plt.xlabel('State')
plt.title('Top 10 States by SBA Loan Default Rate\n')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""FL has by far the highest default rate at 27.4%, nearly 4 points above the next state. This suggests that, during your sample period, SBA‐backed loans in Florida carried substantially more risk than in other large states"""

df.columns

df['NAICS2'] = df['NAICS'].astype(int).astype(str).str[:2]

# Compute counts and default rates by sector
industry_stats = (
    df
      .groupby('NAICS2')['MIS_Status_Binary']
      .agg(loan_count='size', default_rate='mean')
      .reset_index()
)


# Sort and take top 10 riskiest sectors
top10_ind = industry_stats.sort_values('default_rate', ascending=False).head(10)

print("Top 10 NAICS 2-digit Sectors by Default Rate")
print(top10_ind)

# Plot
fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(top10_ind['NAICS2'], top10_ind['default_rate'] * 100, color='teal')

# Label each bar with percent
ax.bar_label(bars, labels=[f"{x:2f}%" for x in top10_ind['default_rate'] * 100], padding=4)

ax.set_ylim(0, top10_ind['default_rate'].max() * 110)
ax.set_ylabel('Default Rate (%)')
ax.set_xlabel('NAICS 2-Digit Sector')
ax.set_title(f'Top 10 SBA Default Rates by Industry Sector\n(NAICS 2-Digits)')
plt.tight_layout()
plt.show()

"""| NAICS 2-Digit | Sector                                      | Loan Count | Default Rate |
| ------------: | ------------------------------------------- | ---------: | -----------: |
|            53 | Real Estate & Rental & Leasing              |     13,632 |        28.7% |
|            52 | Finance & Insurance                         |      9,496 |        28.4% |
|            48 | Transportation & Warehousing                |     20,310 |        26.9% |
|            51 | Information                                 |     11,379 |        24.8% |
|            61 | Educational Services                        |      6,425 |        24.2% |
|            56 | Administrative & Support & Waste Management |     32,685 |        23.6% |
|            45 | Retail Trade                                |     42,514 |        23.4% |
|            23 | Construction                                |     66,646 |        23.3% |
|            49 | Transit & Ground Passenger Transportation   |      2,221 |        23.0% |
|            44 | Retail Trade (second sub-category)          |     84,737 |        22.4% |

"""

df.columns

"""**TRAIN THE MODEL**

# Feature selection and define X and y**
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, make_scorer

# Recompute engineered feature
df['GuaranteeRatio'] = df['SBA_Appv'] / df['GrAppv']

# Check for NoEmployee
df['NoEmp'] = pd.to_numeric(df['NoEmp'], errors='coerce')

# 3. Drop rows where NoEmp or MIS_Status_Binary is missing
df_clean = df[['NoEmp', 'MIS_Status_Binary']].dropna()

# 4. Compute Pearson correlation coefficient between NoEmp and default flag
corr_coef = df_clean['NoEmp'].corr(df_clean['MIS_Status_Binary'])
print(f"Pearson correlation (NoEmp vs. Default): {corr_coef:.4f}")

# 5. Calculate default rate by each distinct employee count
#    (If there are too many distinct employee counts, consider binning instead.)
grouped = df_clean.groupby('NoEmp')['MIS_Status_Binary'].agg(
    loan_count='size',
    default_rate='mean'
).reset_index()

plt.figure(figsize=(8, 5))
plt.scatter(grouped['NoEmp'], grouped['default_rate'], s=20, alpha=0.6)
plt.xlabel('Number of Employees (NoEmp)')
plt.ylabel('Default Rate')
plt.title('Default Rate by Exact Number of Employees')
plt.grid(True)
plt.tight_layout()
plt.show()

# Check for 
df['CreateJob'] = pd.to_numeric(df['CreateJob'], errors='coerce')

# 3. Drop rows where CreateJob or MIS_Status_Binary is missing
df_clean = df[['CreateJob', 'MIS_Status_Binary']].dropna()

# 4. Compute Pearson correlation coefficient between CreateJob and default flag
corr_coef = df_clean['CreateJob'].corr(df_clean['MIS_Status_Binary'])
print(f"Pearson correlation (CreateJob vs. Default): {corr_coef:.4f}")

# 5. Group by each distinct CreateJob value to calculate default rate per job created
grouped = (
    df_clean
    .groupby('CreateJob')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
)

print("\nDefault rate by exact CreateJob count:")
print(grouped.head(10))  # print first 10 rows for inspection

# 6. Plot default rate vs. exact CreateJob count (scatter)
plt.figure(figsize=(8, 5))
plt.scatter(grouped['CreateJob'], grouped['default_rate'], s=20, alpha=0.6)
plt.xlabel('Number of Jobs Created (CreateJob)')
plt.ylabel('Default Rate')
plt.title('Default Rate by Exact CreateJob Count')
plt.grid(True)
plt.tight_layout()
plt.show()

bins = [-1, 0, 5, 10, 20, df_clean['CreateJob'].max()+1]
labels = ['0', '1–5', '6–10', '11–20', '21+']
df_clean['CreateJob_Bin'] = pd.cut(df_clean['CreateJob'], bins=bins, labels=labels, right=False)

grouped = (
    df_clean
    .groupby('CreateJob_Bin')['MIS_Status_Binary']
    .mean()
    .reset_index()
)
print(grouped)
## No significant relationship

# Retained Job
df['RetainedJob'] = pd.to_numeric(df['RetainedJob'], errors='coerce')

# 3. Drop rows where RetainedJob or MIS_Status_Binary is missing
df_clean = df[['RetainedJob', 'MIS_Status_Binary']].dropna()

# 4. Compute Pearson correlation coefficient between RetainedJob and default flag
corr_coef = df_clean['RetainedJob'].corr(df_clean['MIS_Status_Binary'])
print(f"Pearson correlation (RetainedJob vs. Default): {corr_coef:.4f}")

# 5. Group by each distinct RetainedJob value to calculate default rate per job retained
grouped = (
    df_clean
    .groupby('RetainedJob')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
)

print("\nDefault rate by exact RetainedJob count:")
print(grouped.head(10))  # Inspect the first 10 rows

# 6. Plot default rate vs. exact RetainedJob count (scatter)
plt.figure(figsize=(8, 5))
plt.scatter(grouped['RetainedJob'], grouped['default_rate'], s=20, alpha=0.6)
plt.xlabel('Number of Jobs Retained')
plt.ylabel('Default Rate')
plt.title('Default Rate by Exact RetainedJob Count')
plt.grid(True)
plt.tight_layout()
plt.show()

bins = [-1, 0, 5, 10, 20, df_clean['RetainedJob'].max() + 1]
labels = ['0', '1–5', '6–10', '11–20', '21+']
df_clean['RetainedJob_Bin'] = pd.cut(df_clean['RetainedJob'], bins=bins, labels=labels, right=False)

binned = (
    df_clean
    .groupby('RetainedJob_Bin')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
    .dropna()
)

print("\nDefault rate by RetainedJob bins:")
print(binned)

# 8. Plot default rate by RetainedJob bins (bar chart)
plt.figure(figsize=(8, 5))
plt.bar(binned['RetainedJob_Bin'], binned['default_rate'], color='teal', alpha=0.8)
plt.xlabel('Jobs Retained (Binned)')
plt.ylabel('Default Rate')
plt.title('Default Rate by RetainedJob Bins')
plt.ylim(0, binned['default_rate'].max() * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# Pearson correlation (RetainedJob vs. Default): 0.0124 - no significant relationship

# GuaranteeRatio
df_clean = df[['GuaranteeRatio', 'MIS_Status_Binary']].replace([pd.NA, pd.NaT, float('inf'), -float('inf')], pd.NA).dropna()

# 5. Compute Pearson correlation coefficient between GuaranteeRatio and default flag
corr_coef = df_clean['GuaranteeRatio'].corr(df_clean['MIS_Status_Binary'])
print(f"Pearson correlation (GuaranteeRatio vs. Default): {corr_coef:.4f}")

# 6. Group by distinct GuaranteeRatio values to calculate default rate
#    (If there are too many unique ratios, we will bin in the next step)
grouped_exact = (
    df_clean
    .groupby('GuaranteeRatio')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
)

print("\nSample of Default Rate by exact GuaranteeRatio:")
print(grouped_exact.head(10))

# 7. Scatter plot of default rate vs. exact GuaranteeRatio
plt.figure(figsize=(8, 5))
plt.scatter(grouped_exact['GuaranteeRatio'], grouped_exact['default_rate'], s=20, alpha=0.6)
plt.xlabel('GuaranteeRatio (SBA_Appv / GrAppv)')
plt.ylabel('Default Rate')
plt.title('Default Rate by Exact GuaranteeRatio')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Bin GuaranteeRatio into ranges (e.g., 0-0.2, 0.2-0.4, ..., 0.8-1.0, >1.0 if applicable)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, df_clean['GuaranteeRatio'].max()+0.01]
labels = ['0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0', '1.0+']
df_clean['GR_Bin'] = pd.cut(df_clean['GuaranteeRatio'], bins=bins, labels=labels, right=False)

binned = (
    df_clean
    .groupby('GR_Bin')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
    .dropna()
)

print("\nDefault Rate by GuaranteeRatio bins:")
print(binned)

# 9. Bar chart of default rate by GuaranteeRatio bins
plt.figure(figsize=(8, 5))
plt.bar(binned['GR_Bin'], binned['default_rate'], color='mediumseagreen', alpha=0.8)
plt.xlabel('GuaranteeRatio Bins')
plt.ylabel('Default Rate')
plt.title('Default Rate by GuaranteeRatio Bins')
plt.ylim(0, binned['default_rate'].max() * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


## Franchise code
df['FranchiseCode'] = pd.to_numeric(df['FranchiseCode'], errors='coerce')

# 3. Drop rows where FranchiseCode or MIS_Status_Binary is missing
df_clean = df[['FranchiseCode', 'MIS_Status_Binary']].dropna()

# 4. Compute Pearson correlation coefficient between FranchiseCode and default flag
corr_coef = df_clean['FranchiseCode'].corr(df_clean['MIS_Status_Binary'])
print(f"Pearson correlation (FranchiseCode vs. Default): {corr_coef:.4f}")

# 5. Group by each distinct FranchiseCode to calculate loan count and default rate
grouped = (
    df_clean
    .groupby('FranchiseCode')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
)

print("\nDefault rate by exact FranchiseCode (first 10 rows):")
print(grouped.head(10))

# 6. To reduce noise, filter to codes that appear at least N times (e.g., N = 50)
min_loans = 50
filtered = grouped[grouped['loan_count'] >= min_loans]

print(f"\nNumber of distinct FranchiseCodes with ≥ {min_loans} loans: {filtered.shape[0]}")

# 7. Identify top 10 riskiest codes (highest default_rate) among those with sufficient volume
top_risky = filtered.sort_values('default_rate', ascending=False).head(10)
print("\nTop 10 FranchiseCodes by Default Rate (loan_count ≥ 50):")
print(top_risky[['FranchiseCode', 'loan_count', 'default_rate']])

# 8. Identify top 10 safest codes (lowest default_rate) among those with sufficient volume
top_safe = filtered.sort_values('default_rate').head(10)
print("\nTop 10 FranchiseCodes by Lowest Default Rate (loan_count ≥ 50):")
print(top_safe[['FranchiseCode', 'loan_count', 'default_rate']])

# 9. Plot default rate for those top 10 riskiest FranchiseCodes
plt.figure(figsize=(8, 5))
plt.bar(
    top_risky['FranchiseCode'].astype(str),
    top_risky['default_rate'],
    color='indianred',
    alpha=0.8
)
plt.xlabel('FranchiseCode')
plt.ylabel('Default Rate')
plt.title(f'Top 10 Riskiest FranchiseCodes (≥ {min_loans} loans)')
plt.ylim(0, top_risky['default_rate'].max() * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 10. Plot default rate for those top 10 safest FranchiseCodes
plt.figure(figsize=(8, 5))
plt.bar(
    top_safe['FranchiseCode'].astype(str),
    top_safe['default_rate'],
    color='seagreen',
    alpha=0.8
)
plt.xlabel('FranchiseCode')
plt.ylabel('Default Rate')
plt.title(f'Top 10 Safest FranchiseCodes (≥ {min_loans} loans)')
plt.ylim(0, top_safe['default_rate'].max() * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show() ## cungx duocj


# RevLineCr_Binary
df_clean = df[['RevLineCr_Binary', 'MIS_Status_Binary']].dropna()

# 5. Compute Pearson correlation between RevLineCr_Binary and default flag
corr_coef = df_clean['RevLineCr_Binary'].corr(df_clean['MIS_Status_Binary'])
print(f"Pearson correlation (RevLineCr_Binary vs. Default): {corr_coef:.4f}")

# 6. Calculate default rate by RevLineCr category
grouped = (
    df_clean
    .groupby('RevLineCr_Binary')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
)
grouped['category'] = grouped['RevLineCr_Binary'].map({0: 'Term Loan', 1: 'Revolving Line'})

print("\nDefault rate by RevLineCr category:")
print(grouped[['category', 'loan_count', 'default_rate']])

# 7. Plot default rate by RevLineCr category
plt.figure(figsize=(6, 4))
plt.bar(
    grouped['category'],
    grouped['default_rate'],
    color=['skyblue', 'salmon'],
    alpha=0.8
)
plt.xlabel('Loan Type')
plt.ylabel('Default Rate')
plt.title('Default Rate: Term Loan vs. Revolving Line of Credit')
plt.ylim(0, grouped['default_rate'].max() * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# LowDoc Binary
df_clean = df[['LowDoc_Binary', 'MIS_Status_Binary']].dropna()

# 5. Compute Pearson correlation between LowDoc_Binary and default flag
corr_coef = df_clean['LowDoc_Binary'].corr(df_clean['MIS_Status_Binary'])
print(f"Pearson correlation (LowDoc_Binary vs. Default): {corr_coef:.4f}")

# 6. Calculate default rate by LowDoc category
grouped = (
    df_clean
    .groupby('LowDoc_Binary')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
)
grouped['category'] = grouped['LowDoc_Binary'].map({0: 'Standard Documentation', 1: 'Low Documentation'})

print("\nDefault rate by LowDoc category:")
print(grouped[['category', 'loan_count', 'default_rate']])

# 7. Plot default rate by LowDoc category
plt.figure(figsize=(6, 4))
plt.bar(
    grouped['category'],
    grouped['default_rate'],
    color=['steelblue', 'tomato'],
    alpha=0.8
)
plt.xlabel('Documentation Type')
plt.ylabel('Default Rate')
plt.title('Default Rate: Standard vs. Low-Documentation Loans')
plt.ylim(0, grouped['default_rate'].max() * 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Approval FY
df['ApprovalFY'] = pd.to_numeric(df['ApprovalFY'], errors='coerce')
df_year = df[df['ApprovalFY'].notna()].copy()

# Group by Fiscal Year and compute default rate
yearly_stats = (
    df_year
    .groupby('ApprovalFY')['MIS_Status_Binary']
    .agg(loan_count='size', default_rate='mean')
    .reset_index()
)

# Plot default rate over years
plt.figure(figsize=(12, 6))
plt.plot(yearly_stats['ApprovalFY'], yearly_stats['default_rate'], marker='o', linestyle='-')
plt.title('Default Rate by Fiscal Year of Loan Approval')
plt.xlabel('Fiscal Year')
plt.ylabel('Default Rate')
plt.grid(True)
plt.tight_layout()
plt.show()

df['ApprovalFY'] = pd.to_numeric(df['ApprovalFY'], errors='coerce')
print(df['ApprovalFY'].describe())

print("Unique values in ApprovalFY:", df['ApprovalFY'].unique())
print("Minimum ApprovalFY:", df['ApprovalFY'].min())
print("Maximum ApprovalFY:", df['ApprovalFY'].max())

# Check initial number of rows
initial_count = df.shape[0]

# Drop rows with ApprovalFY > 2025
df['ApprovalFY'] = pd.to_numeric(df['ApprovalFY'], errors='coerce')
df_cleaned = df[df['ApprovalFY'] <= 2025]

# Check new number of rows
final_count = df_cleaned.shape[0]

# Print summary
dropped = initial_count - final_count
print(f"Initial rows: {initial_count}")
print(f"Retained rows: {final_count}")
print(f"Dropped rows: {dropped}")

# Initial rows: 899164
# Retained rows: 898993
# Dropped rows: 171

print("Unique values in ApprovalFY:", df_cleaned['ApprovalFY'].unique())
print("Minimum ApprovalFY:", df_cleaned['ApprovalFY'].min())
print("Maximum ApprovalFY:", df_cleaned['ApprovalFY'].max())

# Create InflationRate
df['ApprovalFY_int'] = df['ApprovalFY'].astype(int)


# Minimum ApprovalFY: 1962.0
# Maximum ApprovalFY: 2014.0

inflation_dict = {
    1962: 1.30, 1963: 1.60, 1964: 1.00, 1965: 1.90, 1966: 3.50, 1967: 3.00, 1968: 4.70,
    1969: 6.20, 1970: 5.60, 1971: 3.30, 1972: 3.40, 1973: 8.70, 1974: 12.30, 1975: 6.90,
    1976: 4.90, 1977: 6.70, 1978: 9.00, 1979: 13.30, 1980: 12.50, 1981: 8.90, 1982: 3.80,
    1983: 3.80, 1984: 3.90, 1985: 3.80, 1986: 1.10, 1987: 4.40, 1988: 4.40, 1989: 4.60,
    1990: 6.10, 1991: 3.10, 1992: 2.90, 1993: 2.70, 1994: 2.70, 1995: 2.50, 1996: 3.30,
    1997: 1.70, 1998: 1.60, 1999: 2.70, 2000: 3.40, 2001: 1.60, 2002: 2.40, 2003: 1.90,
    2004: 3.30, 2005: 3.40, 2006: 2.50, 2007: 4.10, 2008: 0.10, 2009: 2.70, 2010: 1.50,
    2011: 3.00, 2012: 1.70, 2013: 1.50, 2014: 0.80
}



df_cleaned['InflationRate'] = df_cleaned['ApprovalFY'].map(inflation_dict)

print(df_cleaned[['ApprovalFY', 'InflationRate']].dropna().head())

# GDP Growth
gdp_growth = {
    1962: 6.10,
    1963: 4.40,
    1964: 5.80,
    1965: 6.40,
    1966: 6.50,
    1967: 2.50,
    1968: 4.80,
    1969: 3.10,
    1970: 0.22,
    1971: 3.29,
    1972: 5.26,
    1973: 5.65,
    1974: -0.54,
    1975: -0.21,
    1976: 5.39,
    1977: 4.62,
    1978: 5.54,
    1979: 3.17,
    1980: -0.26,
    1981: 2.54,
    1982: -1.80,
    1983: 4.58,
    1984: 7.24,
    1985: 4.17,
    1986: 3.46,
    1987: 3.45,
    1988: 4.18,
    1989: 3.67,
    1990: 1.89,
    1991: -0.11,
    1992: 3.52,
    1993: 2.75,
    1994: 4.03,
    1995: 2.68,
    1996: 3.77,
    1997: 4.45,
    1998: 4.48,
    1999: 4.79,
    2000: 4.08,
    2001: 0.96,
    2002: 1.70,
    2003: 2.80,
    2004: 3.85,
    2005: 3.48,
    2006: 2.78,
    2007: 2.00,
    2008: 0.11,
    2009: -2.58,
    2010: 2.70,
    2011: 1.56,
    2012: 2.29,
    2013: 2.12,
    2014: 2.52,
}

df_cleaned['GDPGrowth'] = df_cleaned['ApprovalFY'].astype(int).map(gdp_growth)


# Interest rate
interest_rate = {
    1961: 3.11,
    1962: 3.22,
    1963: 3.37,
    1964: 2.95,
    1965: 2.57,
    1966: 2.65,
    1967: 2.41,
    1968: 1.86,
    1969: 2.85,
    1970: 2.51,
    1971: 0.62,
    1972: 0.88,
    1973: 2.41,
    1974: 1.65,
    1975: -1.28,
    1976: 1.27,
    1977: 0.58,
    1978: 1.89,
    1979: 4.03,
    1980: 5.72,
    1981: 8.59,
    1982: 8.18,
    1983: 6.62,
    1984: 8.14,
    1985: 6.56,
    1986: 6.19,
    1987: 5.59,
    1988: 5.59,
    1989: 6.69,
    1990: 6.04,
    1991: 4.92,
    1992: 3.88,
    1993: 3.55,
    1994: 4.90,
    1995: 6.59,
    1996: 6.32,
    1997: 6.60,
    1998: 7.15,
    1999: 6.49,
    2000: 6.81,
    2001: 4.57,
    2002: 3.07,
    2003: 2.11,
    2004: 1.61,
    2005: 2.96,
    2006: 4.73,
    2007: 5.20,
    2008: 3.10,
    2009: 2.62,
    2010: 2.01,
    2011: 1.16,
    2012: 1.36,
    2013: 1.52,
    2014: 1.48,
}

df_cleaned['InterestRate'] = df_cleaned['ApprovalFY'].astype(int).map(interest_rate)

# Define the final feature set (drop everything else)
features = [
    'Term',
    'DisbursementGross',
    'GuaranteeRatio',
    'FranchiseCode',
    'RevLineCr_Binary',
    'LowDoc_Binary',
    'UrbanRural_Numeric',
    'RealEstate',
    'Recession',
    'NAICS2',
    'State',
    'ApprovalFY', 
    'InterestRate',
    'GDPGrowth',
    'InflationRate']

target = 'MIS_Status_Binary'

# Create modeling DataFrame
df_model = df_cleaned[features + [target]].copy()

# Drop rows with NA in features or target
df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
df_model.dropna(inplace=True)

# Identify categorical columns (to be one-hot encoded)
categorical_cols = [
    'FranchiseCode',
    'RevLineCr_Binary',
    'LowDoc_Binary',
    'UrbanRural_Numeric',
    'RealEstate',
    'Recession',
    'NAICS2',
    'State',
    'ApprovalFY'
]

df_model[categorical_cols] = df_model[categorical_cols].astype(str)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# Split into features (X) and target (y)
X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.9,
    random_state=1)

print("X_train.shape:", X_train.shape)  
print("X_test.shape: ", X_test.shape)  

numeric_cols = [
    'Term',
    'DisbursementGross',
    'GuaranteeRatio',
    'InflationRate',
    'InterestRate',
    'GDPGrowth']

all_cols = X_train.columns.tolist()
categorical_dummy_cols = [c for c in all_cols if c not in numeric_cols]

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('passthrough', 'passthrough', categorical_dummy_cols)],
    remainder='drop')

# -------------------------------
# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from scipy.stats import randint

rf_pipe = Pipeline([
    # 'preproc' will standardize numeric_cols and passthrough one-hot dummies
    ('preproc', preprocessor),
    ('rf', RandomForestClassifier(random_state=42))
])

# ------------------------------------------------------------------------------------------------
# 4) DEFINE PARAMETER DISTRIBUTION FOR RandomizedSearchCV
# ------------------------------------------------------------------------------------------------
param_dist_rf = {
    'rf__n_estimators':      randint(50, 300),
    'rf__max_depth':         randint(5, 50),
    'rf__min_samples_split': randint(2, 20),
    'rf__min_samples_leaf':  randint(1, 10),
    'rf__max_features':      ['sqrt', 'log2', None]
}

# ------------------------------------------------------------------------------------------------
# 5) CONFIGURE RandomizedSearchCV (using ROC‐AUC as the scoring metric)
# ------------------------------------------------------------------------------------------------
search_rf = RandomizedSearchCV(
    estimator=rf_pipe,
    param_distributions=param_dist_rf,
    n_iter=50,           # 50 random combinations
    scoring='roc_auc',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=1,            # set to 1 to avoid parallel‐processing pickling issues
    refit=True
)

# ------------------------------------------------------------------------------------------------
# 6) FIT search_rf ON THE TRAINING SET
# ------------------------------------------------------------------------------------------------
search_rf.fit(X_train, y_train)

# After fitting, the best‐found pipeline is in search_rf.best_estimator_
best_rf = search_rf.best_estimator_

# Print out the best hyperparameters and the best CV ROC‐AUC
print("\n=== Best Hyperparameters ===")
print(search_rf.best_params_)
print("Best CV ROC AUC: ", search_rf.best_score_)

# ------------------------------------------------------------------------------------------------
# 7) EVALUATE ON THE TEST SET (threshold = 0.5)
# ------------------------------------------------------------------------------------------------
# 7.1) Predict labels and probabilities on X_test
y_pred_rf_default = best_rf.predict(X_test)
probs_rf_default  = best_rf.predict_proba(X_test)[:, 1]  # probability of class=1 = Default

print("\n--- Classification Report (Random Forest, threshold=0.5) ---")
print(classification_report(y_test, y_pred_rf_default, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_default))
print("Test ROC AUC: ", roc_auc_score(y_test, probs_rf_default))

# 7.2) Plot ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, probs_rf_default)
plt.figure(figsize=(6,4))
plt.plot(fpr_rf, tpr_rf, color='navy',
         label=f'Random Forest (AUC = {roc_auc_score(y_test, probs_rf_default):.4f})')
plt.plot([0,1], [0,1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------------------
# 8) DEFINE NET PROFIT FUNCTION AND THRESHOLD‐TUNING FUNCTION
# ------------------------------------------------------------------------------------------------
def net_profit(y_true, y_pred, disbursement):
    """
    y_true[i]:       0 = Paid in Full (PIF), 1 = Default (CHGOFF)
    y_pred[i]:       0 = Grant loan, 1 = Deny loan
    disbursement[i]: raw DisbursementGross amount

    Profit logic:
      • If we GRANT (y_pred = 0) and truth = 0 (PIF) → +5% * amt
      • If we GRANT (y_pred = 0) and truth = 1 (Default) → −25% * amt
      • If we DENY  (y_pred = 1) → 0 (no gain/loss)
    """
    profit = 0.0
    for truth, pred, amt in zip(y_true, y_pred, disbursement):
        if pred == 0:  # Grant loan
            if truth == 0:
                profit += 0.05 * amt
            else:
                profit -= 0.25 * amt
        else:  # pred == 1 (Deny)
            if truth == 0:
                profit -= 0.05 * amt   # chi phí cơ hội: lẽ ra có +5% thì bây giờ mất đi
            # nếu truth == 1 và pred == 1, profit += 0 (đúng vừa tránh lỗ)
    return profit

def tune_threshold(y_true, prob_default, disbursement, plot=True):
    """
    Scan thresholds from 0.00 to 0.99 (step 0.01), compute net profit at each, and return 
    the threshold that maximizes net profit.

    Returns:
      best_threshold (float), max_profit (float)
    """
    thresholds = np.arange(0.00, 1.00, 0.01)
    profits = []

    for t in thresholds:
        # If P(Default) >= t → we DENY (pred = 1); else we GRANT (pred = 0)
        y_pred = (prob_default >= t).astype(int)
        profits.append(net_profit(y_true, y_pred, disbursement))

    profits = np.array(profits)
    best_idx = np.nanargmax(profits)
    best_threshold = thresholds[best_idx]
    max_profit = profits[best_idx]

    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(thresholds, profits, marker='o', label="Net Profit")
        plt.axvline(x=best_threshold, color='r', linestyle='--',
                    label=f'Best Threshold = {best_threshold:.2f}')
        plt.title("Threshold vs Net Profit (Random Forest)")
        plt.xlabel("Threshold for P(Default)")
        plt.ylabel("Total Net Profit (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_threshold, max_profit

# ------------------------------------------------------------------------------------------------
# 9) EXTRACT RAW disbursement AMOUNTS FOR THE TEST SET
# ------------------------------------------------------------------------------------------------
# X_test is the ONE-HOT–ENCODED & SCALED matrix, so we need the original 'DisbursementGross' from df_model.
# Since train_test_split preserved the index, we can do:
disb_test = df_model.loc[X_test.index, 'DisbursementGross'].values

# ------------------------------------------------------------------------------------------------
# 10) COMPUTE NET PROFIT AT THRESHOLD = 0.5
# ------------------------------------------------------------------------------------------------
y_pred_rf_05 = (probs_rf_default >= 0.5).astype(int)
profit_rf_05 = net_profit(y_test.values, y_pred_rf_05, disb_test)
print(f"\nNet Profit at threshold = 0.5: ${profit_rf_05:,.2f}")

# ------------------------------------------------------------------------------------------------
# 11) FIND THE THRESHOLD THAT MAXIMIZES NET PROFIT
# ------------------------------------------------------------------------------------------------
best_thresh_rf, best_profit_rf = tune_threshold(
    y_true=y_test.values,
    prob_default=probs_rf_default,
    disbursement=disb_test,
    plot=True
)
print(f"\nOptimal Threshold for Random Forest: {best_thresh_rf:.2f}")
print(f"Maximum Expected Net Profit (Random Forest): ${best_profit_rf:,.2f}")

# ------------------------------------------------------------------------------------------------
# 12) EVALUATE FINAL CLASSIFICATION AT THE OPTIMAL THRESHOLD
# ------------------------------------------------------------------------------------------------
y_pred_rf_opt = (probs_rf_default >= best_thresh_rf).astype(int)

print("\n--- Classification Report with Optimized Threshold (Random Forest) ---")
print(classification_report(y_test, y_pred_rf_opt, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_opt))

# Compute the overall rate of return:
total_disb_test = disb_test.sum()
rate_of_return  = best_profit_rf / total_disb_test * 100

print(f"\nTotal Disbursement (test): ${total_disb_test:,.0f} USD")
print(f"Net Profit       (RF): ${best_profit_rf:,.2f} USD")
print(f"Rate of Return       : {rate_of_return:.2f} %")

import joblib

# Suppose `best_rf` is your fitted Pipeline (preprocessor + RandomForestClassifier).
# For example, after RandomizedSearchCV:
#    best_rf = search_rf.best_estimator_

# 1) SAVE TO DISK
joblib.dump(best_rf, 'rf_pipeline.joblib')
# This writes a file named "rf_pipeline.joblib" in your working directory.

# 2) LATER, LOAD IT BACK INTO MEMORY
loaded_rf = joblib.load('rf_pipeline.joblib')
y_hat_load = loaded_rf.predict(X_test)
print('Loaded model accuracy: ', (y_hat_load == y_test).mean())









# -------------------
probs_default_val = best_rf.predict_proba(X_test)[:, 1]  # Xác suất default

df_val = pd.DataFrame({
    'y_true':        y_test.values,
    'p_default':     probs_default_val,
    'disbursement':  df_model.loc[X_test.index, 'DisbursementGross'].values
})


# 3) TÍNH XÁC SUẤT THÀNH CÔNG (P(success) = 1 - P(default))
df_val['p_success'] = 1.0 - df_val['p_default']

# 4) TÍNH “LỢI NHUẬN NẾU CẤP VỐN” CHO MỖI LOAN
#    • Nếu grant (cấp vốn) mà loan PIF (y_true = 0) → +5% * disbursement
#    • Nếu grant mà loan Default (y_true = 1) → -25% * disbursement
#    (Nếu deny, lợi nhuận = 0)
df_val['profit_if_funded'] = df_val.apply(
    lambda row: 0.05 * row['disbursement'] if row['y_true'] == 0 else -0.25 * row['disbursement'],
    axis=1
)

# 5) SẮP XẾP CÁC LOAN THEO p_success GIẢM DẦN (ÍT RỦI RO NHẤT TRƯỚC)
df_val_sorted = df_val.sort_values(by='p_success', ascending=False).reset_index(drop=True)

# 6) TÍNH CÁC CHỈ SỐ TÍCH LŨY
#    • cumulative_profit: lợi nhuận cộng dồn khi cấp vốn theo thứ tự
#    • index: thứ tự (1, 2, 3, …)
#    • cumulative_fraction: tỉ lệ (%) đã cấp vốn = index / tổng loan
df_val_sorted['cumulative_profit'] = df_val_sorted['profit_if_funded'].cumsum()
df_val_sorted['index'] = np.arange(1, len(df_val_sorted) + 1)
df_val_sorted['cumulative_fraction'] = df_val_sorted['index'] / len(df_val_sorted)

# 7) XÁC ĐỊNH ĐIỂM ĐẠT LỢI NHUẬN TỐI ĐA
best_idx = df_val_sorted['cumulative_profit'].idxmax()
best_fraction = df_val_sorted.loc[best_idx, 'cumulative_fraction']
best_cumulative_profit = df_val_sorted.loc[best_idx, 'cumulative_profit']
best_p_success_cutoff = df_val_sorted.loc[best_idx, 'p_success']

# 8) TÍNH LIFT: CUMULATIVE AVERAGE PROFIT / BASELINE AVERAGE PROFIT
baseline_avg_profit = df_val_sorted['profit_if_funded'].mean()
df_val_sorted['cumulative_avg_profit'] = df_val_sorted['cumulative_profit'] / df_val_sorted['index']
df_val_sorted['profit_lift'] = df_val_sorted['cumulative_avg_profit'] / baseline_avg_profit

# 9) VẼ GAINS CHART: CUMULATIVE PROFIT vs CUMULATIVE FRACTION
plt.figure(figsize=(8, 5))
plt.plot(df_val_sorted['cumulative_fraction'], df_val_sorted['cumulative_profit'],
         marker='o', linewidth=1, label='Cumulative Profit')
plt.axvline(x=best_fraction, color='r', linestyle='--',
            label=f'Best Fraction ≈ {best_fraction:.2f}')
plt.title('Gains Chart: Cumulative Net Profit vs Fraction of Loans Funded')
plt.xlabel('Fraction of Loans Funded (Least to Most Risky)')
plt.ylabel('Cumulative Net Profit (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10) VẼ LIFT CHART: PROFIT LIFT vs CUMULATIVE FRACTION
plt.figure(figsize=(8, 5))
plt.plot(df_val_sorted['cumulative_fraction'], df_val_sorted['profit_lift'],
         marker='o', linewidth=1, label='Profit Lift')
plt.axvline(x=best_fraction, color='r', linestyle='--',
            label=f'Best Fraction ≈ {best_fraction:.2f}')
plt.title('Lift Chart: Profit Lift vs Fraction of Loans Funded')
plt.xlabel('Fraction of Loans Funded (Least to Most Risky)')
plt.ylabel('Profit Lift (Cumulative Avg / Baseline Avg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 11) IN KẾT QUẢ CHÍNH:
#    a) Điểm lý tưởng (best_fraction) cho đến bao nhiêu phần trăm của validation 
#       để đạt lợi nhuận tối đa
#    b) Ngưỡng “probability of success” (p_success) tương ứng
result_summary = pd.DataFrame({
    'Best Fraction of Loans to Fund (validation)': [best_fraction],
    'Maximum Cumulative Profit (USD)':               [best_cumulative_profit],
    'P(success) Cutoff to Fund':                    [best_p_success_cutoff]
})

print(result_summary.to_string(index=False))







































