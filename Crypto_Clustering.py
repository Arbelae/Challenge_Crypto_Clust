#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    "crypto_market_data.csv",
    index_col="coin_id")

# Display sample data
df_market_data.head(3)


# In[3]:


# Generate summary statistics
df_market_data.describe()


# In[4]:


# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data

# In[5]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
market_data_scaled = StandardScaler().fit_transform(df_market_data[['price_change_percentage_24h', 'price_change_percentage_7d',
       'price_change_percentage_14d', 'price_change_percentage_30d',
       'price_change_percentage_60d', 'price_change_percentage_200d',
       'price_change_percentage_1y']])


# In[6]:


# Create a DataFrame with the scaled data
# Copy the crypto names from the original data
df_market_data_transformed = pd.DataFrame(
    market_data_scaled, 
    columns =['price_change_percentage_24h', 'price_change_percentage_7d',
       'price_change_percentage_14d', 'price_change_percentage_30d',
       'price_change_percentage_60d', 'price_change_percentage_200d',
       'price_change_percentage_1y']
)

# Set the coinid column as index
df_market_data_transformed["coin_id"]=df_market_data.index
df_market_data_transformed = df_market_data_transformed.set_index("coin_id")

# Display sample data
df_market_data_transformed.head()


# ---

# ### Find the Best Value for k Using the Original Data.

# In[7]:


# Create a list with the number of k-values from 1 to 11

k = list(range(1,11))


# In[8]:


# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list

for i in k:
        k_model = KMeans(n_clusters=i, random_state=0)
        k_model.fit(df_market_data_transformed)
        inertia.append(k_model.inertia_)
        


# In[9]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k":k, "inertia":inertia}


# Create a DataFrame with the data to plot the Elbow curve
de_elbow = pd.DataFrame(elbow_data)
de_elbow.head()


# In[10]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
de_elbow.hvplot.line(
    x="k",
    y= "inertia",
    title="Optimal value for k",
    xticks=k,
    rot=90
)


# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# **Answer:**    3 - 4

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Data

# In[11]:


# Initialise the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=3)


# In[12]:


# Fit the K-Means model using the scaled data
model.fit(df_market_data_transformed)


# In[13]:


# Predict the clusters to group the cryptocurrencies using the scaled data
k_4=model.predict(df_market_data_transformed)

# Print the resulting array of cluster values.
print(k_4)


# In[14]:


# Create a copy of the DataFrame
df_market_data_transformed_predictions = df_market_data_transformed.copy()


# In[15]:


# Add a new column to the DataFrame with the predicted clusters
df_market_data_transformed_predictions['Predicted_Clusters']=k_4

# Display sample data
df_market_data_transformed_predictions.tail(5)


# In[16]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Colour the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_market_data_transformed_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
   by="Predicted_Clusters"
)


# ---

# ### Optimise Clusters with Principal Component Analysis.

# In[17]:


# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)


# In[18]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
Info_pca = pca.fit_transform(df_market_data_transformed)

# View the first five rows of the DataFrame. 
Info_pca[:5]


# In[19]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
vr= pca.explained_variance_ratio_
vr


# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** ( 0.3719856 +  0.34700813 + 0.17603793)*100
#                              Appox  89.50 %

# In[20]:


# Create a new DataFrame with the PCA data.
# Creating a DataFrame with the PCA data
Info_pca_df = pd.DataFrame(Info_pca, columns=['PCA1','PCA2','PCA3'])

# Copy the crypto names from the original data
# Set the coinid column as index
Info_pca_df["coin_id"]=df_market_data_transformed.index
Info_pca_df = Info_pca_df.set_index("coin_id")

# Display sample data
Info_pca_df.head(5)


# ---

# ### Find the Best Value for k Using the PCA Data

# In[21]:


# Create a list with the number of k-values from 1 to 11
k=list(range(1,11))


# In[22]:


# Create an empty list to store the inertia values
inertia=[]

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=1)
    k_model.fit(Info_pca_df)
    inertia.append(k_model.inertia_)


# In[23]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k":k, "inertia":inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)
df_elbow.head()


# In[24]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
dataPlot=de_elbow.hvplot.line(
    x="k",
    y= "inertia",
    title="Optimal value for k",
    xticks=k,
    rot=90
)
dataPlot


# #### Answer the following questions: 
# 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:**   3 - 4
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:**  its not different from the original but maybe more accurate as it shows more close to 4

# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# In[25]:


# Initialise the K-Means model using the best value for k
model = KMeans(n_clusters=3, random_state=0)


# In[26]:


# Fit the K-Means model using the PCA data
model.fit(Info_pca_df)


# In[27]:


# Predict the clusters to group the cryptocurrencies using the PCA data
k_3PCA=model.predict(Info_pca_df)

# Print the resulting array of cluster values.
k_3PCA


# In[28]:


# Create a copy of the DataFrame with the PCA data
PCA_predictions = Info_pca_df.copy()

# Add a new column to the DataFrame with the predicted clusters
PCA_predictions['Predicted_Clusters']=k_3PCA

# Display sample data
PCA_predictions.tail(10)


# In[29]:


# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Colour the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
PCA_predictions.hvplot.scatter(
    x="PCA1",
    y="PCA2",
   by="Predicted_Clusters"
)


# ### Visualise and Compare the Results
# 
# In this section, you will visually analyse the cluster analysis results by contrasting the outcome with and without using the optimisation techniques.

# In[ ]:


# Composite plot to contrast the Elbow curves
(de_elbow + dataPlot).cols(1)


# In[ ]:


# Composite plot to contrast the clusters
(df_market_data_transformed_predictions + PCA_predictions).cols(1)


# #### Answer the following question: 
# 
#   * **Question:** After visually analysing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** We actually see that is not need to use alot of features to get similar performance
