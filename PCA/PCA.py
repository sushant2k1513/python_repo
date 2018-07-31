# Package import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Import data
from sklearn.datasets import load_breast_cancer
can = load_breast_cancer()

# Make a data frame
df = pd.DataFrame(can['data'], columns = can['feature_names'])
df.head(10)
df.shape

# Normalize the features
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(df)
scaled_df = scalar.transform(df)

# Get PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit(scaled_df)
df_pca = pca.transform(scaled_df)
df_pca.shape

# Plot
plt.figure(figsize =(8, 6)) 
plt.scatter(df_pca[:, 0], df_pca[:, 1], c = can['target'], cmap ='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
pca.components_

# Heat MAP
df_comp = pd.DataFrame(pca.components_, columns = can['feature_names'])
plt.figure(figsize =(14, 6))
sns.heatmap(df_comp)
