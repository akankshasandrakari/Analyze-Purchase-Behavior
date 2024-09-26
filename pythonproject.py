import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 
# Load data 
data = pd.read_csv('C:/Users/akank/OneDrive/Desktop/DEP/purchase_data.csv') 
# Data Preprocessing 
data['Purchase Date'] = pd.to_datetime(data['Purchase Date']) 
data['Total Amount'] = data['Quantity'] * data['Price'] 
# Exploratory Data Analysis 
print(data.describe()) 
print(data['Product ID'].value_counts()) 
# Visualization 
plt.figure(figsize=(12, 6)) 
sns.histplot(data['Total Amount'], bins=30, kde=True) 
plt.title('Distribution of Purchase Amounts') 
plt.xlabel('Amount') 
plt.ylabel('Frequency') 
plt.show() 
# Clustering Example (K-means) 
customer_data = data[['Customer ID', 'Total Amount']] 
kmeans = KMeans(n_clusters=3) 
data['Cluster'] = kmeans.fit_predict(customer_data) 
# Plot Clusters 
plt.figure(figsize=(12, 6)) 
sns.scatterplot(x='Total Amount', y='Customer ID', hue='Cluster', data=data, palette='viridis') 
plt.title('Customer Segmentation') 
plt.xlabel('Total Amount') 
plt.ylabel('Customer ID') 
plt.show()
