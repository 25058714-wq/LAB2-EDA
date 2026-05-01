import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Font Settings ==========
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ========== Read Data ==========
df = pd.read_csv("WorldEnergy.csv")

# ========== ASEAN 6 Countries (Data Available) ==========
asean = [
    "Indonesia", "Thailand", "Malaysia", "Philippines", "Singapore",
    "Vietnam"
]

# ========== Select Columns ==========
cols = [
    "country", "year",
    "coal_consumption", "oil_consumption", "gas_consumption",
    "renewables_consumption", "primary_energy_consumption"
]
df = df[cols].copy()

# ========== Filter & Clean Data ==========
df = df[df["country"].isin(asean)]
df = df[(df["year"] >= 2000) & (df["year"] <= 2023)]

# ==========================
# ✅ 显示清洗前数据量
# ==========================
print("="*60)
print("BEFORE DATA CLEANING (Rows, Columns):", df.shape)
print("BEFORE - Missing values:\n", df.isnull().sum())
print("="*60)

# 清洗
df = df.dropna()
df = df[df["primary_energy_consumption"] > 0]

# ==========================
# ✅ 显示清洗后数据量
# ==========================
print("\nAFTER DATA CLEANING (Rows, Columns):", df.shape)
print("AFTER - Missing values:\n", df.isnull().sum())
print("="*60)

# ========== Calculate Ratios ==========
df["coal_ratio"] = df["coal_consumption"] / df["primary_energy_consumption"]
df["oil_ratio"] = df["oil_consumption"] / df["primary_energy_consumption"]
df["gas_ratio"] = df["gas_consumption"] / df["primary_energy_consumption"]
df["renew_ratio"] = df["renewables_consumption"] / df["primary_energy_consumption"]

# ------------------------------
# Figure 1: ASEAN Energy Structure Trend (Stack Plot)
# ------------------------------
year_mean = df.groupby("year").mean(numeric_only=True)
plt.figure(figsize=(12,5))
plt.stackplot(year_mean.index, 
              year_mean["coal_ratio"],
              year_mean["oil_ratio"],
              year_mean["gas_ratio"],
              year_mean["renew_ratio"],
              labels=["Coal","Oil","Natural Gas","Clean Energy"],
              colors=["brown","orange","red","green"])
plt.title("ASEAN Energy Structure Change (2000-2023)")
plt.xlabel("Year")
plt.ylabel("Ratio")
plt.legend()
plt.show()

# ------------------------------
# Figure 2: Clean Energy Ratio Trend
# ------------------------------
plt.figure(figsize=(10,4))
plt.plot(year_mean.index, year_mean["renew_ratio"], marker="o", color="green", linewidth=3)
plt.title("Trend of ASEAN Clean Energy Ratio")
plt.grid(True)
plt.show()

# ------------------------------
# Figure 3: ASEAN Energy Structure in 2023 (Pie Chart)
# ------------------------------
latest_yr = df[df["year"] == 2023].mean(numeric_only=True)
plt.figure(figsize=(6,6))
plt.pie([latest_yr["coal_ratio"], latest_yr["oil_ratio"], latest_yr["gas_ratio"], latest_yr["renew_ratio"]],
        labels=["Coal","Oil","Natural Gas","Clean Energy"],
        autopct="%.1f%%",
        colors=["brown","orange","red","green"])
plt.title("ASEAN Overall Energy Structure in 2023")
plt.show()

# ------------------------------
# Figure 4: Average Clean Energy Ratio by Country
# ------------------------------
c = df.groupby("country")["renew_ratio"].mean().sort_values()
plt.figure(figsize=(11,5))
c.plot(kind="bar", color="green")
plt.title("Average Clean Energy Ratio by Country")
plt.xticks(rotation=30)
plt.show()

# ------------------------------
# Figure 5: Average Coal Ratio by Country
# ------------------------------
c2 = df.groupby("country")["coal_ratio"].mean().sort_values()
plt.figure(figsize=(11,5))
c2.plot(kind="bar", color="brown")
plt.title("Average Coal Ratio by Country")
plt.xticks(rotation=30)
plt.show()

# ------------------------------
# Figure 6: Energy Structure of ASEAN Countries in 2023 (Stacked Bar)
# ------------------------------
latest = df[df["year"] == 2023].groupby("country").mean(numeric_only=True)
latest[["coal_ratio","oil_ratio","gas_ratio","renew_ratio"]].plot(kind="bar", stacked=True, figsize=(13,6))
plt.title("Energy Structure of ASEAN Countries in 2023")
plt.ylabel("Ratio")
plt.legend(["Coal","Oil","Natural Gas","Clean Energy"])
plt.show()

# ------------------------------
# Figure 7: Distribution of Clean Energy Ratio
# ------------------------------
plt.figure(figsize=(9,4))
plt.hist(df["renew_ratio"], bins=12, color="lightgreen", edgecolor="black")
plt.title("Distribution of Clean Energy Ratio in ASEAN")
plt.show()

# ------------------------------
# Figure 8: Boxplot of Clean Energy Ratio by Country
# ------------------------------
plt.figure(figsize=(12,5))
sns.boxplot(x="country", y="renew_ratio", data=df)
plt.title("Clean Energy Ratio by Country")
plt.xticks(rotation=45)
plt.show()

# ------------------------------
# Figure 9: Correlation between Clean Energy and Coal Ratio
# ------------------------------
plt.figure(figsize=(9,5))
sns.scatterplot(x="coal_ratio", y="renew_ratio", hue="country", data=df)
plt.title("Correlation between Clean Energy Ratio and Coal Ratio")
plt.show()

# ------------------------------
# Figure 10: Correlation Heatmap of Energy Structure
# ------------------------------
corr = df[["coal_ratio","oil_ratio","gas_ratio","renew_ratio"]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Energy Structure Indicators")
plt.show()
