import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os

# =========================
# IRIS DATASET VISUALIZATION
# =========================

iris = load_iris()

X = iris.data
y = iris.target

# Scatter Plot: Petal Length vs Petal Width
for i, name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 2], X[y == i, 3], label=name)

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Width by Species')
plt.legend()
plt.show()

# Histogram: Sepal Quality
sepal_quality = X[:, 0] / X[:, 1]

plt.hist(sepal_quality, bins=20, edgecolor='black')
plt.xlabel('Sepal Quality (Length / Width)')
plt.ylabel('Frequency')
plt.title('Distribution of Sepal Quality')
plt.show()

# Box Plot: Sepal Length
plt.boxplot(X[:, 0], vert=False)
plt.xlabel('Sepal Length')
plt.title('Box Plot of Sepal Length')
plt.show()


# =========================
# LOAN DATASET VISUALIZATION
# =========================

loan_data = pd.read_csv(
    os.path.expanduser(r"~\Downloads\LoanDataset - LoansDatasest.csv"),
    sep=None,
    engine="python",
    encoding="utf-8-sig"
)

# Clean column names
loan_data.columns = loan_data.columns.str.strip()


# =========================
# DATA CLEANING
# =========================

# Create a completely clean numeric column for loan amount
loan_data["loan_amount_clean"] = (
    loan_data["loan_amnt"]
    .astype(str)
    .str.replace(r"[^0-9.]", "", regex=True)
)

loan_data["loan_amount_clean"] = pd.to_numeric(
    loan_data["loan_amount_clean"], errors="coerce"
)

# Convert other columns
loan_data["loan_int_rate"] = pd.to_numeric(
    loan_data["loan_int_rate"], errors="coerce"
)

loan_data["customer_age"] = pd.to_numeric(
    loan_data["customer_age"], errors="coerce"
)

loan_data["employment_duration"] = pd.to_numeric(
    loan_data["employment_duration"], errors="coerce"
)


# =========================
# DEBUG CHECKS
# =========================

print("\nColumn Types:")
print(loan_data.dtypes)

print("\nLoan Amount Examples:")
print(loan_data[["loan_amnt", "loan_amount_clean"]].head())

print("\nStatistics:")
print(
    loan_data[
        ["loan_amount_clean", "loan_int_rate", "customer_age", "employment_duration"]
    ].describe()
)


# =========================
# PREPARE CLEAN DATASETS
# =========================

scatter_data = loan_data.dropna(
    subset=["loan_amount_clean", "loan_int_rate"]
)

line_data = loan_data.dropna(
    subset=["loan_amount_clean", "customer_age"]
)

bar_data = loan_data.dropna(
    subset=["loan_amount_clean", "employment_duration"]
)


# =========================
# VISUALIZATIONS
# =========================

# Scatter Plot: Loan Amount vs Interest Rate
if len(scatter_data) > 0:

    plt.scatter(
        scatter_data["loan_amount_clean"],
        scatter_data["loan_int_rate"],
        alpha=0.6
    )

    plt.xlabel("Loan Amount")
    plt.ylabel("Interest Rate")
    plt.title("Loan Amount vs Interest Rate")
    plt.show()

else:
    print("No data for scatter plot")
#The scatter plot shows a positive correlation between loan amount and interest rate, indicating that larger loans tend to have higher interest rates, which may reflect lenders' risk assessment practices.

# Line Plot: Customer Age vs Loan Amount
if len(line_data) > 0:

    sorted_data = line_data.sort_values("customer_age")

    plt.plot(
        sorted_data["customer_age"],
        sorted_data["loan_amount_clean"],
        marker="o",
        linestyle="-",
        alpha=0.7
    )

    plt.xlabel("Customer Age")
    plt.ylabel("Loan Amount")
    plt.title("Customer Age vs Loan Amount")
    plt.show()

else:
    print("No data for line plot")
# Line plot shows a general upward trend, indicating that older customers tend to receive larger loan amounts, which may reflect lenders' perceptions of financial stability with age.

# Bar Chart: Employment Duration vs Average Loan Amount
if len(bar_data) > 0:

    employment_avg = (
        bar_data
        .groupby("employment_duration")["loan_amount_clean"]
        .mean()
    )

    plt.bar(
        employment_avg.index,
        employment_avg.values
    )

    plt.xlabel("Employment Duration (Years)")
    plt.ylabel("Average Loan Amount")
    plt.title("Average Loan Amount by Employment Duration")
    plt.xticks(rotation=45)
    plt.show()
#The bar chart shows a clear trend of increasing average loan amounts with longer employment durations, suggesting that lenders may view longer employment as a positive factor in loan approval and amount decisions.