
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:\\Users\\DELL\\Downloads\\archive (4)\\community_health_evaluation_dataset.csv')



# 1. Data Cleaning
print("Missing Values:\n", data.isnull().sum())
print("\nDuplicates:", data.duplicated().sum())

# 2. Descriptive Statistics
print("\nSummary Statistics:\n", data.describe())

# 3. Exploratory Data Analysis (EDA)

# Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], bins=15, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Socioeconomic Status (SES) Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='SES', data=data, palette='viridis', hue=None, legend=False)
plt.title('Socioeconomic Status (SES) Distribution')
plt.xlabel('SES')
plt.ylabel('Count')
plt.show()

# Patient Satisfaction by Service Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Service Type', y='Patient Satisfaction (1-10)', data=data, palette='coolwarm', hue=None, legend=False)
plt.title('Patient Satisfaction by Service Type')
plt.xlabel('Service Type')
plt.ylabel('Satisfaction Score')
plt.show()

# Step Frequency vs. Joint Angle
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Step Frequency (steps/min)', y='Joint Angle (°)', hue='Service Type', data=data, palette='Set2')
plt.title('Step Frequency vs Joint Angle')
plt.xlabel('Step Frequency (steps/min)')
plt.ylabel('Joint Angle (°)')
plt.legend(title='Service Type')
plt.show()

# Correlation Matrix for Numerical Variables
plt.figure(figsize=(10, 8))

# Select only numeric columns for correlation matrix calculation
numeric_data = data.select_dtypes(include=[np.number])

# Calculate correlation and print it to verify
correlation = numeric_data.corr()
print("Correlation Matrix:\n", correlation)  # Print to verify

# Plot the heatmap if correlation matrix exists
if not correlation.empty:
    sns.heatmap(correlation, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
else:
    print("No numeric columns for correlation matrix.")

# Quality of Life Score by Age Group
age_bins = [18, 30, 45, 60, 75]
age_labels = ['18-30', '31-45', '46-60', '61-75']
data['Age Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Age Group', y='Quality of Life Score', data=data, palette='pastel')
plt.title('Quality of Life Score by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Quality of Life Score')
plt.show()
