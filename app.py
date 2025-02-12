import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the data set
df = pd.read_csv('test_data.csv')
print(df.head(5))


# Heatmap data corelation 
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=2)
plt.title("Heatmap of Data Correlation")
plt.show()


# bar chart for survied or not
plt.figure(figsize=(6,4))
df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.title("Survival Count")
plt.xticks([0, 1], ["Not Survived", "Survived"], rotation=0)
plt.show()

# histogram for age in titanicship passengers
plt.figure(figsize=(6,4))
plt.hist(df['Age'], bins=10, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()

# pie chart for passenger class 1 or not
plt.figure(figsize=(6,6))
df['Pclass_1'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title("Passenger Class Distribution")
plt.ylabel("") 
plt.show()

