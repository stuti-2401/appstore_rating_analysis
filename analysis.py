import matplotlib.pyplot as plt
import seaborn as sns
from data import load_and_clean_data  

sns.set(style="whitegrid")  

df = load_and_clean_data()

# 1. Distribution of App Ratings
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Apps')
plt.tight_layout()
plt.show()

# 2. Free vs Paid Apps Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Type', palette='Set2')
plt.title('Free vs Paid Apps')
plt.xlabel('Type')
plt.ylabel('Number of Apps')
plt.tight_layout()
plt.show()

# 3. Top 15 Categories by Average Rating
plt.figure(figsize=(10, 6))
category_rating = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(15)
sns.barplot(x=category_rating.values, y=category_rating.index, palette='viridis')
plt.title('Top 15 Categories by Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('App Category')
plt.tight_layout()
plt.show()

# 4. Installs vs Rating (Scatter Plot, log scale for Installs)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Installs', y='Rating', alpha=0.6)
plt.title('Installs vs Rating')
plt.xlabel('Number of Installs')
plt.ylabel('Rating')
plt.xscale('log')  
plt.tight_layout()
plt.show()
