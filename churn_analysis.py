import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Load customer dataset
files = glob.glob("*.csv")
if not files:
    raise FileNotFoundError("Upload the customer churn CSV file first.")

df = pd.read_csv(files[0])
df.columns = df.columns.str.strip()

print("Dataset loaded:", files[0])
print("Rows and columns:", df.shape)
print("Available columns:", list(df.columns))

# Find columns automatically
def find_col(names):
    for name in names:
        for col in df.columns:
            if col.lower().replace("", "").replace(" ", "") == name.replace("", "").replace(" ", ""):
                return col
    return None

churn_col = find_col(["Churn", "Exited", "Attrition", "Customer_Status"])
tenure_col = find_col(["tenure", "Tenure_Months"])
monthly_col = find_col(["MonthlyCharges", "Monthly_Charges", "MonthlyRevenue"])
contract_col = find_col(["Contract", "ContractType"])
payment_col = find_col(["PaymentMethod", "Payment_Method"])

if not churn_col:
    raise ValueError("Churn column not found. Check the printed column names.")

# Clean churn values
def is_churned(value):
    return str(value).strip().lower() in ["yes", "true", "1", "churned", "exited"]

df["Churn_Flag"] = df[churn_col].apply(is_churned)

# Calculate churn rate
churn_rate = df["Churn_Flag"].mean() * 100
print(f"\nCustomer Churn Rate: {churn_rate:.2f}%")

# Calculate CLV estimate
if monthly_col and tenure_col:
    df[monthly_col] = pd.to_numeric(df[monthly_col], errors="coerce")
    df[tenure_col] = pd.to_numeric(df[tenure_col], errors="coerce")
    df["Estimated_CLV"] = df[monthly_col] * df[tenure_col]
    average_clv = df["Estimated_CLV"].mean()
    print(f"Average Estimated CLV: {average_clv:.2f}")
else:
    average_clv = None
    print("CLV unavailable: monthly charges or tenure column missing.")

# Calculate MRR loss
if monthly_col:
    mrr_loss = df.loc[df["Churn_Flag"], monthly_col].sum()
    print(f"Monthly Recurring Revenue Loss: {mrr_loss:.2f}")
else:
    mrr_loss = None
    print("MRR loss unavailable: monthly charges column missing.")

# Churn by contract
if contract_col:
    contract_churn = df.groupby(contract_col)["Churn_Flag"].mean() * 100
    contract_churn.plot(kind="bar", title="Churn Rate by Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.tight_layout()
    plt.savefig("contract_churn.png")
    plt.close()

# Churn by tenure
if tenure_col:
    df["Tenure_Group"] = pd.cut(
        df[tenure_col],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12", "13-24", "25-48", "49+"]
    )
    (df.groupby("Tenure_Group", observed=False)["Churn_Flag"].mean() * 100).plot(
        kind="bar", title="Churn Rate by Tenure"
    )
    plt.ylabel("Churn Rate (%)")
    plt.tight_layout()
    plt.savefig("tenure_churn.png")
    plt.close()

# Churn by payment method
if payment_col:
    (df.groupby(payment_col)["Churn_Flag"].mean() * 100).plot(
        kind="bar", title="Churn Rate by Payment Method"
    )
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("payment_churn.png")
    plt.close()

# Correlation heatmap
numeric = df.select_dtypes(include="number")
if len(numeric.columns) > 1:
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Customer Data Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

# Save analysis report
with open("churn_report.txt", "w") as report:
    report.write("CUSTOMER CHURN ANALYSIS REPORT\n")
    report.write("=" * 35 + "\n")
    report.write(f"Total Customers: {len(df)}\n")
    report.write(f"Churned Customers: {df['Churn_Flag'].sum()}\n")
    report.write(f"Customer Churn Rate: {churn_rate:.2f}%\n")
    report.write(f"Average Estimated CLV: {average_clv}\n")
    report.write(f"Monthly Recurring Revenue Loss: {mrr_loss}\n")
    report.write("\nRECOMMENDATIONS\n")
    report.write("1. Improve customer support and satisfaction.\n")
    report.write("2. Offer loyalty rewards and retention discounts.\n")
    report.write("3. Identify customers at high risk of leaving.\n")
    report.write("4. Encourage long-term subscription contracts.\n")
    report.write("5. Monitor payment-related customer issues.\n")

print("\nAnalysis completed. Reports and graphs saved.")
