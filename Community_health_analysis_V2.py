import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/content/community_health_evaluation_dataset.csv")

# ----------------------
# 1. Basic Preprocessing
# ----------------------

# Drop Participant ID (not useful for analysis)
df = df.drop(columns=["Participant ID"])

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# ----------------------
# 2. Exploratory Data Analysis (EDA)
# ----------------------

# Summary statistics
print("\nSummary:\n", df.describe(include="all"))

# Age distribution
plt.figure(figsize=(7,4))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution of Participants")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Gender distribution
plt.figure(figsize=(5,4))
sns.countplot(x="Gender", data=df, palette="Set2")
plt.title("Gender Distribution")
plt.show()

# Service Type vs Visit Frequency
plt.figure(figsize=(8,5))
sns.countplot(x="Service Type", hue="Visit Frequency", data=df)
plt.title("Service Type vs Visit Frequency")
plt.show()

# ----------------------
# 3. Biomechanical Measures
# ----------------------

# Step Frequency vs Stride Length
plt.figure(figsize=(7,5))
sns.scatterplot(x="Step Frequency (steps/min)", y="Stride Length (m)", hue="Gender", data=df)
plt.title("Step Frequency vs Stride Length")
plt.show()

# Joint Angle distribution by EMG Activity
plt.figure(figsize=(7,5))
sns.boxplot(x="EMG Activity", y="Joint Angle (°)", data=df, palette="coolwarm")
plt.title("Joint Angle by EMG Activity Level")
plt.show()

# ----------------------
# 4. Outcomes Analysis
# ----------------------

# Patient Satisfaction distribution
plt.figure(figsize=(7,4))
sns.histplot(df['Patient Satisfaction (1-10)'], bins=10, kde=False, color="orange")
plt.title("Patient Satisfaction Distribution")
plt.xlabel("Satisfaction Score (1-10)")
plt.ylabel("Count")
plt.show()

# Quality of Life by Service Type
plt.figure(figsize=(7,5))
sns.boxplot(x="Service Type", y="Quality of Life Score", data=df, palette="Set3")
plt.title("Quality of Life Score by Service Type")
plt.show()


plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()
