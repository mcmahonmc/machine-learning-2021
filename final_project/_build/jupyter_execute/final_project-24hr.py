#!/usr/bin/env python
# coding: utf-8

# ## Using mean hourly activity to give summarized 24 hr period

# In[1]:


import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
from matplotlib.dates import WeekdayLocator
import plotly.express as px
import seaborn as sns

from math import ceil
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics, svm, manifold
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# ## Load data

# In[2]:


actdf = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/actigraphy_data_24hrday_df.csv')
print('actigraphy df')
print(actdf.shape)

targets = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/target_data.csv')
targets = targets.set_index('subject')
targets['edge_mean'] = targets[[x for x in targets.columns if x.startswith('edge_')]].mean(axis=1)
print('targets')
print(targets.shape)

targets[:5]


# In[3]:


x = StandardScaler().fit_transform(actdf.T.values) # normalizing the features


# In[4]:


sns.heatmap(actdf.T, cmap="YlGnBu")


# In[5]:


sns.heatmap(x, cmap="YlGnBu")


# In[6]:


sns.lineplot(x=range(0,len(x[100])), y=x[100])


# In[7]:


drop_subs = [ subject for subject in actdf.columns if int(subject) not in targets.index.values ]
drop_subs_idx = [ actdf.columns.get_loc(subject) for subject in actdf.columns if int(subject) not in targets.index.values ]

actdf = actdf.drop(drop_subs, axis=1)[:-3]
x = np.delete(x, drop_subs_idx, axis=0)

print(actdf.shape)
print(x.shape)


# In[8]:


np.mean(x),np.std(x)


# ## Load standard rest-activity measures
# 

# In[9]:


rar = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/rar_df.csv', index_col=0)

drop_subs = [ int(subject) for subject in rar.index if str(subject) not in actdf.columns.values ]
drop_subs

rar2 = (rar[~rar.index.isin(drop_subs)])
print(rar2.shape)


# In[10]:


[col for col in targets.columns if 'mean_active' in col]


# In[11]:


[col for col in targets.columns if 'mean_sleep' in col]


# In[12]:


rar2 = rar2.merge(targets[['total_ac_mean_active',
                    'duration_mean_sleep', 
                    'total_ac_mean_sleep', 
                   'efficiency_mean_sleep',
                   'sleep_time_mean_sleep',
                   'sleep_time_sd_sleep',
                   'onset_latency_mean_sleep']],
          left_index=True, right_index=True)

sns.heatmap(rar2.isnull(), cmap='terrain')


# ### Define targets
# 
# target data uses output from rar dataframe merged with other variables of interest

# **Missing data**
# 
# Missing data here for CESD (this is intended for young adults only), GDS (this is intended for older adults only), and some of the MRI measures (due to poor image quality).
# 
# For targets of interest, will impute missing values with the mean.

# In[13]:


sns.heatmap(targets.isnull(), cmap='terrain')


# In[14]:


# targets.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final/target_data.csv', index=True)


# # Dimensionality Reduction
# 
# ## PCA

# In[15]:


pca = PCA()
data_pcs = pca.fit_transform(x)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlim(0,20)
print(pca.explained_variance_[:8])


# In[16]:


components_n = 3


# In[17]:


pca = PCA(n_components=components_n)

# X: Xarray-like of shape (n_samples, n_features)
pca.fit(x)
manifold_2Da_pca = pca.fit_transform(x)
manifold_2D_pca = pd.DataFrame(manifold_2Da_pca, 
                               columns=['Component %s' % (i+1) for i in range(0, len(manifold_2Da_pca[0]))])

# Left with 2 dimensions
print(manifold_2D_pca.shape)
manifold_2D_pca.head()


# In[18]:


print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# ## Isomap
# 
# [Benalexkeen resource](https://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/)

# In[19]:


iso = manifold.Isomap(n_neighbors=6, n_components=components_n)
iso.fit(x)
manifold_2Da = iso.transform(x)
manifold_2D_iso = pd.DataFrame(manifold_2Da, 
                               columns=['Component %s' % (i+1) for i in range(0, len(manifold_2Da[0]))])


# Left with 2 dimensions
print(manifold_2D_iso.shape)
manifold_2D_iso.head()


# ## LLE

# In[20]:


lle = manifold.LocallyLinearEmbedding(n_neighbors=6, n_components=components_n)
lle.fit(x)
manifold_2Da_lle = lle.transform(x)
manifold_2D_lle = pd.DataFrame(manifold_2Da, 
                               columns=['Component %s' % (i+1) for i in range(0, len(manifold_2Da[0]))])



# Left with 2 dimensions
print(manifold_2D_lle.shape)
manifold_2D_lle.head()


# # Comparison with traditional sleep and rest-activity measures

# ## PCA
# 
# Component 1 - total activity <br>
# Component 2 - acrophase (phi)

# In[21]:


corr = manifold_2D_pca.join(rar2.reset_index().drop('index', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[22]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_pca['Component 1'] > manifold_2D_pca['Component 1'].median()),
                 np.where(manifold_2D_pca['Component 1'] < manifold_2D_pca['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High PC1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low PC1')


# In[23]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_pca['Component 2'] > manifold_2D_pca['Component 2'].median()),
                 np.where(manifold_2D_pca['Component 2'] < manifold_2D_pca['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High PC2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low PC2')


# In[24]:


comp3_high_idx, comp3_low_idx = [np.where(manifold_2D_pca['Component 3'] > manifold_2D_pca['Component 3'].median()),
                 np.where(manifold_2D_pca['Component 3'] < manifold_2D_pca['Component 3'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp3_high_idx].mean(axis=0), label='High PC2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp3_low_idx].mean(axis=0), label='Low PC2')


# ## Isomap
# 
# Isomap uses the above principle to create a similarity matrix for eigenvalue decomposition. Unlike other non-linear dimensionality reduction like LLE & LPP which only use local information, isomap uses the local information to create a global similarity matrix. The isomap algorithm uses euclidean metrics to prepare the neighborhood graph. Then, it approximates the geodesic distance between two points by measuring shortest path between these points using graph distance. Thus, it approximates both global as well as the local structure of the dataset in the low dimensional embedding. -[Paperspace Blog](https://blog.paperspace.com/dimension-reduction-with-isomap/)
# 
# Component 2 - phi

# In[25]:


corr = manifold_2D_iso.join(rar2.reset_index().drop('index', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[26]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_iso['Component 1'] > manifold_2D_iso['Component 1'].median()),
                 np.where(manifold_2D_iso['Component 1'] < manifold_2D_iso['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High C1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low C1')


# In[27]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_iso['Component 2'] > manifold_2D_iso['Component 2'].median()),
                 np.where(manifold_2D_iso['Component 2'] < manifold_2D_iso['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low C2')


# In[28]:


comp3_high_idx, comp3_low_idx = [np.where(manifold_2D_iso['Component 3'] > manifold_2D_iso['Component 3'].median()),
                 np.where(manifold_2D_iso['Component 3'] < manifold_2D_iso['Component 3'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp3_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp3_low_idx].mean(axis=0), label='Low C2')


# ## LLE
# 
# Component 2 - phi

# In[29]:


corr = manifold_2D_lle.join(rar2.reset_index().drop('index', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[30]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_lle['Component 1'] > manifold_2D_lle['Component 1'].median()),
                 np.where(manifold_2D_lle['Component 1'] < manifold_2D_lle['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High C1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low C1')


# In[31]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_lle['Component 2'] > manifold_2D_lle['Component 2'].median()),
                 np.where(manifold_2D_lle['Component 2'] < manifold_2D_lle['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low C2')


# In[32]:


comp3_high_idx, comp3_low_idx = [np.where(manifold_2D_lle['Component 3'] > manifold_2D_lle['Component 3'].median()),
                 np.where(manifold_2D_lle['Component 3'] < manifold_2D_lle['Component 3'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp3_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp3_low_idx].mean(axis=0), label='Low C2')


# In[ ]:





# ## Correlations
# 
# MAE: Average absolute error between the model prediction and the actual observed data. <br>
# RMSE: Lower the RMSE, the more closely a model is able to predict the actual observations.

# In[33]:


sns.pairplot(data=manifold_2D_iso.join(targets[['trails_b_z_score_x', 'mod_mean', 'edge_mean']].reset_index()), 
             x_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')],
            y_vars = ['trails_b_z_score_x', 'mod_mean', 'edge_mean'])


# In[34]:


#define cross-validation method to use
cv = LeaveOneOut()

#build multiple linear regression model
model = LinearRegression()

#trails b
scores = cross_val_score(model, manifold_2D_iso, targets['trails_b_z_score_x'], 
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

print('Trails B, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))


#modularity
scores = cross_val_score(model, manifold_2D_iso, targets['mod_mean'], 
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

print('Modularity, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))

      
#edge strength
scores = cross_val_score(model, manifold_2D_iso, targets['edge_mean'], 
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)


print('Edge Strength, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))


# # Classification 
# 
# ## SVM
# 
# [Datacamp](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)
# 
# ### Cognition

# In[35]:


targets.columns


# In[36]:


targets = targets.reset_index()

targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] > 1, "High", "Average")
targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] < -1, "Low", targets['trails_b_group'])

average_idx = targets[targets['trails_b_group'] == 'Average'].index.values


# In[37]:


print('PCA Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_pca, targets['trails_b_group'], metric='euclidean'))
print('Isomap Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_iso, targets['trails_b_group'], metric='euclidean'))
print('LLE Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_lle, targets['trails_b_group'], metric='euclidean'))

silscore = pd.DataFrame({'method': ['PCA', 'Isomap', 'LLE'],
                        'silscore': [metrics.silhouette_score(manifold_2D_pca, targets['trails_b_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_iso, targets['trails_b_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_lle, targets['trails_b_group'], metric='euclidean')]})


print('\n\n%s' % silscore.min())


# In[38]:


sns.pairplot(manifold_2D_iso.join(targets['trails_b_group'].reset_index()), 
             hue = 'trails_b_group', palette='Set1',
             x_vars = [col for col in manifold_2D_lle.columns if col.startswith('Component')],
             y_vars = [col for col in manifold_2D_lle.columns if col.startswith('Component')])


# :::{note}
# The Silhouette score measures the separability between clusters based on the distances between and within clusters. It calculates the mean intra-cluster distance (a), which is the mean distance within a cluster, and the mean nearest-cluster distance (b), which is the distance between a sample and the nearest cluster it is not a part of, for each sample. Then, the Silhouette coefficient for a sample is (b - a) / max(a, b). - [Maarten Grootendorst](https://www.maartengrootendorst.com/blog/customer/)
# :::
# 
# Isomap yields the lowest silhouette score, suggesting that this dimensionality reduction technique as implemented with the selected parameters outperformed PCA and LLE techniques in terms of cluster separability based on Trails B performance ('high', 'average', 'low').

# In[39]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(manifold_2D_iso.drop(index=average_idx), 
                                                    targets['trails_b_group'][targets['trails_b_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[40]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']
accdf = pd.DataFrame()

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    accdf = accdf.append({'method': kernel_method,
                      'accuracy': acc}, ignore_index=True)
    

    print(kernel_method + ", Accuracy: %.2f " % acc)
    

print('\n\n%s' % accdf.max())


# In[41]:


clf = svm.SVC(kernel=accdf.max()['method'])

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[42]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[43]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test['Component 1'], 
              y=X_test['Component 2'], 
              z=targets.iloc[y_test.index.values]['trails_b_z_score_x'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'Trails B Z-Score',
                  'color': 'Predicted vs. True Performance'
              },
              title='Components and Trails B Z-Score')


# ### Brain

# In[44]:


targets['mod_mean'].fillna(targets['mod_mean'].mean(), inplace=True)
print(targets['mod_mean'])
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
targets['mod_mean_scaled'] = scaler.fit_transform(targets['mod_mean'].values.reshape(-1,1))
targets['mod_mean_scaled'] 

targets['mod_mean_group'] = np.where(targets['mod_mean_scaled'] > 1, "High", "Average")
targets['mod_mean_group'] = np.where(targets['mod_mean_scaled'] < -1, "Low", targets['mod_mean_group'] )

average_idx = targets[targets['mod_mean_group'] == 'Average'].index.values


# In[45]:


print('PCA Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_pca, targets['mod_mean_group'], metric='euclidean'))
print('Isomap Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_iso, targets['mod_mean_group'], metric='euclidean'))
print('LLE Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_lle, targets['mod_mean_group'], metric='euclidean'))

silscore = pd.DataFrame({'method': ['PCA', 'Isomap', 'LLE'],
                        'silscore': [metrics.silhouette_score(manifold_2D_pca, targets['mod_mean_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_iso, targets['mod_mean_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_lle, targets['mod_mean_group'], metric='euclidean')]})


print('\n\n%s' % silscore.min())


# In[46]:


sns.pairplot(manifold_2D_iso.join(targets['mod_mean_group'].reset_index()), 
             hue = 'mod_mean_group', palette='Set1',
             x_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')],
             y_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')])


# In[47]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(manifold_2D_iso.drop(index=average_idx),
                                                    targets['mod_mean_group'][targets['mod_mean_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[48]:


targets['mod_mean_group'].unique()


# In[49]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']
accdf = pd.DataFrame()

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    accdf = accdf.append({'method': kernel_method,
                      'accuracy': acc}, ignore_index=True)
    

    print(kernel_method + ", Accuracy: %.2f " % acc)
    

print('\n\n%s' % accdf.max())


# In[50]:


#Create a svm Classifier
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
clf = svm.SVC(kernel=accdf.max()['method'])

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[51]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[52]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test['Component 1'], 
              y=X_test['Component 2'], 
              z=targets.iloc[y_test.index.values]['mod_mean_group'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'DMN-FPN Modularity',
                  'color': 'Predicted vs. Truth'
              },
              title='Components and DMN-FPN Modularity')


# In[53]:


targets['edge_mean'] = targets[[x for x in targets.columns if x.startswith('edge_')]].mean(axis=1)


# In[54]:


targets['edge_mean'].fillna(targets['edge_mean'].mean(), inplace=True)
print(targets['edge_mean'])
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
targets['edge_mean_scaled'] = scaler.fit_transform(targets['edge_mean'].values.reshape(-1,1))
targets['edge_mean_scaled'] 

targets['edge_mean_group'] = np.where(targets['edge_mean_scaled'] > 1, "High", "Average")
targets['edge_mean_group'] = np.where(targets['edge_mean_scaled'] < -1, "Low", targets['edge_mean_group'] )

average_idx = targets[targets['edge_mean_group'] == 'Average'].index.values


# In[55]:


print('PCA Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_pca, targets['edge_mean_group'], metric='euclidean'))
print('Isomap Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_iso, targets['edge_mean_group'], metric='euclidean'))
print('LLE Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_lle, targets['edge_mean_group'], metric='euclidean'))

silscore = pd.DataFrame({'method': ['PCA', 'Isomap', 'LLE'],
                        'silscore': [metrics.silhouette_score(manifold_2D_pca, targets['edge_mean_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_iso, targets['edge_mean_group'], metric='euclidean'),
                                    metrics.silhouette_score(manifold_2D_lle, targets['edge_mean_group'], metric='euclidean')]})


print('\n\n%s' % silscore.min())


# In[56]:


sns.pairplot(manifold_2D_lle.join(targets['edge_mean_group'].reset_index()), 
             hue = 'edge_mean_group', palette='Set1',
             x_vars = [col for col in manifold_2D_lle.columns if col.startswith('Component')],
             y_vars = [col for col in manifold_2D_lle.columns if col.startswith('Component')])


# In[57]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(manifold_2D_iso.drop(index=average_idx),
                                                    targets['edge_mean_group'][targets['edge_mean_group'] != 'Average'], 
                                                    test_size=0.3, random_state=100) 

print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
print('Test data points: dim_x = %s, dim_y = %s' % (X_test.shape, y_test.shape))


# In[58]:


targets['edge_mean_group'].unique()


# In[59]:


#Create a svm Classifier
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']
accdf = pd.DataFrame()

for kernel_method in kernel_methods:
    
    clf = svm.SVC(kernel=kernel_method)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    accdf = accdf.append({'method': kernel_method,
                      'accuracy': acc}, ignore_index=True)
    

    print(kernel_method + ", Accuracy: %.2f " % acc)
    

print('\n\n%s' % accdf.max())


# In[60]:


#Create a svm Classifier
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
clf = svm.SVC(kernel=accdf.max()['method'])

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[61]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: %.2f" % metrics.precision_score(y_test, y_pred, average='weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: %.2f" % metrics.recall_score(y_test, y_pred, average='weighted'))


# In[62]:


markers = y_pred + ', ' + y_test

px.scatter_3d(x=X_test['Component 1'], 
              y=X_test['Component 2'], 
              z=targets.iloc[y_test.index.values]['edge_mean_group'],
              color=markers,
              labels = {
                  'x' : 'Component 1',
                  'y' : 'Component 2',
                  'z' : 'Memory Network Edge Strength',
                  'color': 'Predicted vs. Truth'
              },
              title='Components and Memory Network Edge Strength')


# In[ ]:




