#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


df=pd.read_csv("Ev Sales.csv")
df.head()
df.info()


# In[3]:


df.head()


# In[4]:


df.value_counts("YEAR")


# In[5]:


print(df.columns)


# In[6]:


# Step 1 - Apply standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df.iloc[:,1:7] = scaler.fit_transform(df.iloc[:,1:7])


# In[7]:


df.head()


# In[8]:


from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
df['YEAR'] = le.fit_transform(df['YEAR'])


# In[9]:


df.head()


# In[22]:


plt.figure(figsize=(10, 6))
plt.plot(df['YEAR'],df['TOTAL'], marker='o', color='b', label='Total Sales')
plt.title('Monthly Total EV Sales Trend (Apr 2017 - Aug 2017)')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[25]:


plt.figure(figsize=(10, 6))
plt.bar(df['YEAR'], df['2 W'], label='2-Wheeler')
plt.bar(df['YEAR'], df['3 W'], bottom=df['2 W'], label='3-Wheeler')
plt.bar(df['YEAR'], df['4 W'], bottom=df['2 W'] + df['3 W'], label='4-Wheeler')
plt.bar(df['YEAR'], df['BUS'], bottom=df['2 W'] + df['3 W'] + df['4 W'], label='Bus')
plt.title('Monthly EV Sales Breakdown by Segment')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[27]:


plt.figure(figsize=(12, 8))
for col, color in zip(['2 W', '3 W', '4 W', 'BUS'], ['orange', 'green', 'purple', 'red']):
    plt.plot(df['YEAR'], df[col], marker='o', linestyle='-', color=color, label=col)
    plt.title('Monthly Sales Trend by Vehicle Segment')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[13]:


df.info()


# In[28]:


plt.scatter(df['YEAR'],df['BUS'])
plt.xlabel('YEAR')
plt.ylabel('BUS')


# In[16]:


plt.scatter(df['YEAR'],df['2 W'])
plt.xlabel('YEAR')
plt.ylabel('2 W')


# In[17]:


plt.scatter(df['YEAR'],df['3 W'])
plt.xlabel('YEAR')
plt.ylabel('3 W')


# In[18]:


plt.scatter(df['YEAR'],df['4 W'])
plt.xlabel('YEAR')
plt.ylabel('4 W')


# In[ ]:


from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# Function to calculate the stability across bootstrapped samples using ARI
def calculate_ari_stability(data, k_range, n_rep=10, n_boot=100):
    ari_results = {k: [] for k in k_range}

    for k in k_range:
        # Run bootstrapping for each value of k
        for _ in range(n_boot):
            # Resample the data
            boot_data = resample(data, random_state=1234)

            # Run k-means on original and bootstrapped samples
            kmeans_original = KMeans(n_clusters=k, n_init=n_rep, random_state=1234).fit(data)
            kmeans_boot = KMeans(n_clusters=k, n_init=n_rep, random_state=1234).fit(boot_data)

            # Calculate ARI between original and bootstrapped sample clustering
            ari = adjusted_rand_score(kmeans_original.labels_, kmeans_boot.labels_)
            ari_results[k].append(ari)

    # Calculate average ARI for each k
    avg_ari = {k: np.mean(ari_results[k]) for k in k_range}

    return avg_ari

# Perform the ARI stability analysis
k_values = range(2, 9)
ari_stability = calculate_ari_stability(df, k_values)

# Plot the adjusted Rand index for each k
plt.plot(list(ari_stability.keys()), list(ari_stability.values()), marker='o')
plt.xlabel("Number of Segments")
plt.ylabel("Adjusted Rand Index")
plt.title("Cluster Stability by Number of Segments")
plt.grid()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Assume we want to plot the distribution of distances from cluster 4's center
# Extract the labels and focus on cluster 4
cluster_4_data =df[best_kmeans.labels_ == 3]  # Cluster "4" in R corresponds to index 3 in Python (0-based indexing)

# Calculate distances of points in cluster 4 to its centroid
cluster_4_center = best_kmeans.cluster_centers_[3]
distances = np.linalg.norm(cluster_4_data - cluster_4_center, axis=1)

# Plot the histogram with limits from 0 to 1
plt.hist(distances, bins=10, range=(0, 1), color='grey', edgecolor='black')
plt.xlabel("Distance from Cluster Center")
plt.ylabel("Frequency")
plt.title("Histogram of Distances for Cluster 4")
plt.xlim(0, 1)
plt.grid()
plt.show()


# In[ ]:




