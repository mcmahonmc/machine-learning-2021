#!/usr/bin/env python
# coding: utf-8

# # Data-driven assessment of rest-activity patterns
# 
# PSY 394S Machine Learning <br>
# Megan McMahon <br>
# Fall 2021 <br>

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

import warnings
warnings.filterwarnings('ignore')


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
print(targets.shape)


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


components_n = 2


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
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title='PCA Component Correlations with Standard RAR and Sleep Measures')


# In[22]:


# correlations with psqi components
corr = manifold_2D_pca.join(targets[[col for col in targets.columns if col.startswith('component_')] + ['global_psqi']].reset_index().drop('subject', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title = 'PCA Component Correlations with PSQI')


# In[23]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_pca['Component 1'] > manifold_2D_pca['Component 1'].median()),
                 np.where(manifold_2D_pca['Component 1'] < manifold_2D_pca['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High PC1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low PC1')


# In[24]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_pca['Component 2'] > manifold_2D_pca['Component 2'].median()),
                 np.where(manifold_2D_pca['Component 2'] < manifold_2D_pca['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High PC2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low PC2')


# ## Isomap
# 
# Isomap uses the above principle to create a similarity matrix for eigenvalue decomposition. Unlike other non-linear dimensionality reduction like LLE & LPP which only use local information, isomap uses the local information to create a global similarity matrix. The isomap algorithm uses euclidean metrics to prepare the neighborhood graph. Then, it approximates the geodesic distance between two points by measuring shortest path between these points using graph distance. Thus, it approximates both global as well as the local structure of the dataset in the low dimensional embedding. -[Paperspace Blog](https://blog.paperspace.com/dimension-reduction-with-isomap/)
# 
# Component 2 - phi

# In[25]:


corr = manifold_2D_iso.join(rar2.reset_index().drop('index', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[26]:


# correlations with psqi components
corr = manifold_2D_iso.join(targets[[col for col in targets.columns if col.startswith('component_')] + ['global_psqi']].reset_index().drop('subject', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title = 'Isomap Component Correlations with PSQI')


# In[27]:


sns.scatterplot(data = manifold_2D_iso.join(targets[[col for col in targets.columns if 'z_score' in col] +
                                   [col for col in targets.columns if 'zscore' in col]].reset_index()),
                                            x='Component 2', y='trails_b_z_score_x')


# In[28]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_iso['Component 1'] > manifold_2D_iso['Component 1'].median()),
                 np.where(manifold_2D_iso['Component 1'] < manifold_2D_iso['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High C1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low C1')


# In[29]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_iso['Component 2'] > manifold_2D_iso['Component 2'].median()),
                 np.where(manifold_2D_iso['Component 2'] < manifold_2D_iso['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low C2')


# ## LLE
# 
# Component 2 - phi

# In[30]:


corr = manifold_2D_lle.join(rar2.reset_index().drop('index', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[31]:


# correlations with psqi components
corr = manifold_2D_lle.join(targets[[col for col in targets.columns if col.startswith('component_')] + ['global_psqi']].reset_index().drop('subject', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title = 'LLE Component Correlations with PSQI')


# In[32]:


comp1_high_idx, comp1_low_idx = [np.where(manifold_2D_lle['Component 1'] > manifold_2D_lle['Component 1'].median()),
                 np.where(manifold_2D_lle['Component 1'] < manifold_2D_lle['Component 1'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp1_high_idx].mean(axis=0), label='High C1')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp1_low_idx].mean(axis=0), label='Low C1')


# In[33]:


comp2_high_idx, comp2_low_idx = [np.where(manifold_2D_lle['Component 2'] > manifold_2D_lle['Component 2'].median()),
                 np.where(manifold_2D_lle['Component 2'] < manifold_2D_lle['Component 2'].median())]

sns.lineplot(x=range(0,len(x[0])), y=x[comp2_high_idx].mean(axis=0), label='High C2')
sns.lineplot(x=range(0,len(x[0])), y=x.mean(axis=0), label='Mean')
sns.lineplot(x=range(0,len(x[0])), y=x[comp2_low_idx].mean(axis=0), label='Low C2')


# In[ ]:





# ## Correlations
# 
# MAE: Average absolute error between the model prediction and the actual observed data. <br>
# RMSE: Lower the RMSE, the more closely a model is able to predict the actual observations.

# In[34]:


targets = targets.reset_index()


# In[35]:


corr = manifold_2D_pca.join(targets[[col for col in targets.columns if 'z_score' in col] +
                                   [col for col in targets.columns if 'zscore' in col] + ['pvt_rt_mean', 'pvt_fs']].reset_index().drop('index', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title = 'PCA Component Correlations with Cognitive Measures')


# In[36]:


corr = manifold_2D_iso.join(targets[[col for col in targets.columns if 'z_score' in col] +
                                   [col for col in targets.columns if 'zscore' in col] + ['pvt_rt_mean', 'pvt_fs']].reset_index().drop('index', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title = 'Isomap Component Correlations with Cognitive Measures')


# In[37]:


corr = manifold_2D_lle.join(targets[[col for col in targets.columns if 'z_score' in col] +
                                   [col for col in targets.columns if 'zscore' in col] + ['pvt_rt_mean', 'pvt_fs']].reset_index().drop('index', axis=1)).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > .2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title = 'LLE Component Correlations with Cognitive Measures')


# In[38]:


sns.pairplot(data=manifold_2D_iso.join(targets[['cvlt_ldelay_recall_zscore', 'trails_b_z_score_x', 'Left.Hippocampus', 'cc_fa', 'edge_mean']].reset_index()), 
             y_vars = [col for col in manifold_2D_iso.columns if col.startswith('Component')],
            x_vars = ['cvlt_ldelay_recall_zscore', 'trails_b_z_score_x', 'Left.Hippocampus', 'cc_fa', 'edge_mean'])


# In[39]:


#define cross-validation method to use
cv = LeaveOneOut()

#build multiple linear regression model
model = LinearRegression()



# #cvlt l delay standard RAR
print('standard RAR and sleep')
cvdf = rar2.reset_index()[['amp']].join(targets['cvlt_ldelay_recall_zscore']).dropna()
scores = cross_val_score(model, 
                         cvdf[['amp']], 
                         cvdf['cvlt_ldelay_recall_zscore'],
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

print('CVLT L Delay Recall, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))

# #cvlt l delay
print('PCA components')
cvdf = manifold_2D_pca.join(targets['cvlt_ldelay_recall_zscore']).dropna(subset=['cvlt_ldelay_recall_zscore'])
scores = cross_val_score(model, 
                         cvdf[[col for col in cvdf.columns if col.startswith('Component ')]], 
                         cvdf['cvlt_ldelay_recall_zscore'],
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

print('CVLT L Delay Recall, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))


# #cvlt l delay
print('PCA component 1')
cvdf = manifold_2D_pca.join(targets['cvlt_ldelay_recall_zscore']).dropna()
scores = cross_val_score(model, 
                         cvdf[['Component 1']],
                         cvdf['cvlt_ldelay_recall_zscore'],
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

print('CVLT L Delay Recall, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))

# #cvlt l delay
print('PCA component 2')
cvdf = manifold_2D_pca.join(targets['cvlt_ldelay_recall_zscore']).dropna()
scores = cross_val_score(model, 
                         cvdf[['Component 2']],
                         cvdf['cvlt_ldelay_recall_zscore'],
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

print('CVLT L Delay Recall, MAE: %.2f, RMSE: %.2f' % (np.mean(np.absolute(scores)), np.sqrt(np.mean(np.absolute(scores)))))


# In[40]:


cvdf[['Component 1', 'cvlt_ldelay_recall_zscore']].corr()


# # Classification 
# 
# ## SVM
# 
# [Datacamp](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)

# In[41]:


logdf=pd.DataFrame()

def run_svm_classifier(manifold_2D_pca, manifold_2D_iso, manifold_2D_lle, 
                       target_group, target_measure, target_label, test_n=.3):
    
    manifold_2D_pca = manifold_2D_pca.reset_index(drop=True)
    manifold_2D_iso = manifold_2D_iso.reset_index(drop=True)
    manifold_2D_lle = manifold_2D_lle.reset_index(drop=True)
    
    target_group = target_group.reset_index(drop=True)
    target_measure = target_measure.reset_index(drop=True)
        
    # calculate Silhouette score
    print('PCA Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_pca, target_group, metric='euclidean'))

    print('Isomap Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_iso, target_group, metric='euclidean'))

    print('LLE Silhouette score: %.3f' % metrics.silhouette_score(manifold_2D_lle, target_group, metric='euclidean'))

    silscore = pd.DataFrame({'method': ['pca', 'iso', 'lle'],
                            'silscore': [metrics.silhouette_score(manifold_2D_pca, target_group, metric='euclidean'),
                                        metrics.silhouette_score(manifold_2D_iso, target_group, metric='euclidean'),
                                        metrics.silhouette_score(manifold_2D_lle, target_group, metric='euclidean')]})


    print('\n\n%s' % silscore['method'][silscore['silscore'].argmin()])

    manifold_2D=eval('manifold_2D_' + silscore['method'][silscore['silscore'].argmin()])
    sns.pairplot((manifold_2D).join(target_measure).join(target_group),
             hue=target_group.name, palette='Set1',
             x_vars = [col for col in manifold_2D.columns if col.startswith('Component')],
             y_vars = [col for col in manifold_2D.columns if col.startswith('Component')])
    plt.show()

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(manifold_2D, 
                                                        target_group,
                                                        test_size=test_n, random_state=10) 

    print('Training data points: dim_x = %s, dim_y = %s' % (X_train.shape, y_train.shape))
    print('Test data points: dim_x = %s, dim_y = %s \n\n' % (X_test.shape, y_test.shape))

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
        

    print(accdf)

    clf = svm.SVC(kernel=accdf['method'][accdf['accuracy'].argmax()])

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    markers = y_pred + ', ' + y_test

    fig = px.scatter_3d(x=X_test['Component 1'], 
                  y=X_test['Component 2'], 
                  z=target_measure[y_test.index.values],
                  color=markers,
                  labels = {
                      'x' : 'Component 1',
                      'y' : 'Component 2',
                      'z' : target_label,
                      'color': 'Predicted vs. Truth'
                  },
                  title='Components and %s' % target_label)

    fig.show()

    log_dict = {'target': target_label,
                      'manifold_method': silscore['method'][silscore['silscore'].argmin()],
                     'silhouette_score': silscore['silscore'][silscore['silscore'].argmin()],
                     'svm_kernel': accdf['method'][accdf['accuracy'].argmax()],
                     'svm_accuracy': accdf['accuracy'][accdf['accuracy'].argmax()],
                     'svm_precision': metrics.precision_score(y_test, y_pred, average='weighted'),
                     'svm_recall': metrics.recall_score(y_test, y_pred, average='weighted')}
    
    log = pd.DataFrame.from_dict([log_dict])
    
    return log


# ### Cognition
# 
# ### CVLT Long Delay Performance in Older Adults

# In[42]:


na_idx = np.where(targets['cvlt_ldelay_recall_zscore'].isnull())[0]

targets['cvlt_l_group'] = np.where(targets['cvlt_ldelay_recall_zscore'] > np.percentile(targets['cvlt_ldelay_recall_zscore'].dropna(), 70), "High", "Medium")
targets['cvlt_l_group'] = np.where(targets['cvlt_ldelay_recall_zscore'] < np.percentile(targets['cvlt_ldelay_recall_zscore'].dropna(), 30), "Low", targets['cvlt_l_group'])
targets['cvlt_l_group'][na_idx] = np.nan

med_idx = targets[targets['cvlt_l_group'] == 'Medium'].index.values
drop_idx = list(na_idx) + list(med_idx)

sns.boxplot(x=targets['cvlt_l_group'], 
            y=targets['cvlt_ldelay_recall_zscore'],
           palette='Set1')


# :::{note}
# The Silhouette score measures the separability between clusters based on the distances between and within clusters. It calculates the mean intra-cluster distance (a), which is the mean distance within a cluster, and the mean nearest-cluster distance (b), which is the distance between a sample and the nearest cluster it is not a part of, for each sample. Then, the Silhouette coefficient for a sample is (b - a) / max(a, b). - [Maarten Grootendorst](https://www.maartengrootendorst.com/blog/customer/)
# :::
# 

# In[43]:


log = run_svm_classifier(manifold_2D_pca.drop(index=drop_idx), 
                   manifold_2D_iso.drop(index=drop_idx), 
                   manifold_2D_lle.drop(index=drop_idx), 
                   targets['cvlt_l_group'].drop(index=drop_idx), 
                   targets['cvlt_ldelay_recall_zscore'].drop(index=drop_idx), 
                   'CVLT Long Delay')

logdf = logdf.append(log, ignore_index=True)
log


# ### Trails B Performance in YA and OA

# In[44]:


na_idx = np.where(targets['trails_b_z_score_x'].isnull())[0]

targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] > 1, "High", "Average")
targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] < -1, "Low", targets['trails_b_group'])

average_idx = targets[targets['trails_b_group'] == 'Average'].index.values
drop_idx = list(na_idx) + list(average_idx)

sns.boxplot(x=targets['trails_b_group'], 
            y=targets['trails_b_z_score_x'],
           palette='Set1', order = ['Low', 'Average', 'High'])


# In[45]:


log = run_svm_classifier(manifold_2D_pca.drop(index=drop_idx), 
                   manifold_2D_iso.drop(index=drop_idx), 
                   manifold_2D_lle.drop(index=drop_idx), 
                   targets['trails_b_group'].drop(index=drop_idx), 
                   targets['trails_b_z_score_x'].drop(index=drop_idx), 
                   'Trails B (Population)')

logdf = logdf.append(log, ignore_index=True)
log


# In[46]:


na_idx = np.where(targets['trails_b_z_score_x'].isnull())[0]

targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] > np.percentile(targets['trails_b_z_score_x'].dropna(), 70), "High", "Medium")
targets['trails_b_group'] = np.where(targets['trails_b_z_score_x'] < np.percentile(targets['trails_b_z_score_x'].dropna(), 30), "Low", targets['trails_b_group'])

med_idx = targets[targets['trails_b_group'] == 'Medium'].index.values
drop_idx = list(na_idx) + list(med_idx)

sns.boxplot(x=targets['trails_b_group'], 
            y=targets['trails_b_z_score_x'],
           palette='Set1', order = ['Low', 'Medium', 'High'])


# In[47]:


log = run_svm_classifier(manifold_2D_pca.drop(index=drop_idx), 
                   manifold_2D_iso.drop(index=drop_idx), 
                   manifold_2D_lle.drop(index=drop_idx), 
                   targets['trails_b_group'].drop(index=drop_idx), 
                   targets['trails_b_z_score_x'].drop(index=drop_idx), 
                   'Trails B (Sample)')

logdf = logdf.append(log, ignore_index=True)
log


# ### Brain
# 
# ### Hippocampal Volume

# In[48]:


targets = targets.reset_index()


# In[49]:


na_idx = np.where(targets['Left.Hippocampus'].isnull())[0]

targets['left_hc_group'] = np.where(targets['Left.Hippocampus'] > np.percentile(targets['Left.Hippocampus'].dropna(), 70), "High", "Medium")
targets['left_hc_group'] = np.where(targets['Left.Hippocampus'] < np.percentile(targets['Left.Hippocampus'].dropna(), 30), "Low", targets['left_hc_group'] )

drop_idx = list(targets[targets['left_hc_group'] == 'Medium'].index.values) + list(na_idx)

sns.boxplot(x=targets[targets['group'] == 'Young Adults']['left_hc_group'], 
            y=targets[targets['group'] == 'Young Adults']['Left.Hippocampus'],
           palette='Set1', order = ['Low', 'Medium', 'High'])

sns.boxplot(x=targets[targets['group'] == 'Older Adults']['left_hc_group'], 
            y=targets[targets['group'] == 'Older Adults']['Left.Hippocampus'],
           palette='Set2', order = ['Low', 'Medium', 'High'])


# In[50]:


log = run_svm_classifier(manifold_2D_pca.drop(index=drop_idx), 
                   manifold_2D_iso.drop(index=drop_idx), 
                   manifold_2D_lle.drop(index=drop_idx), 
                   targets['left_hc_group'].drop(index=drop_idx), 
                   targets['Left.Hippocampus'].drop(index=drop_idx), 
                   'Left Hippocampal Volume')

logdf = logdf.append(log, ignore_index=True)
log


# ### CC FA

# In[51]:


#outliers?
from scipy import stats

np.where(stats.zscore(targets['cc_fa'].dropna()) > 3)


# In[52]:


na_idx = np.where(targets['cc_fa'].isnull())[0]

targets['cc_fa_group'] = np.where(targets['cc_fa'] > np.percentile(targets['cc_fa'].dropna(), 70), "High", "Medium")
targets['cc_fa_group'] = np.where(targets['cc_fa'] < np.percentile(targets['cc_fa'].dropna(), 30), "Low", targets['cc_fa_group'] )

drop_idx = list(targets[targets['cc_fa_group'] == 'Medium'].index.values) + list(na_idx)

sns.boxplot(x=targets[targets['group'] == 'Young Adults']['cc_fa_group'], 
            y=targets[targets['group'] == 'Young Adults']['cc_fa'],
           palette='Set1', order = ['Low', 'Medium', 'High'])

sns.boxplot(x=targets[targets['group'] == 'Older Adults']['cc_fa_group'], 
            y=targets[targets['group'] == 'Older Adults']['cc_fa'],
           palette='Set2', order = ['Low', 'Medium', 'High'])


# In[53]:


log = run_svm_classifier(manifold_2D_pca.drop(index=drop_idx), 
                   manifold_2D_iso.drop(index=drop_idx), 
                   manifold_2D_lle.drop(index=drop_idx), 
                   targets['cc_fa_group'].drop(index=drop_idx), 
                   targets['cc_fa'].drop(index=drop_idx), 
                   'Corpus Callosum FA')

logdf = logdf.append(log, ignore_index=True)
log


# ### Retrieval network edge strength

# In[54]:


targets['edge_mean'].fillna(targets['edge_mean'].mean(), inplace=True)
print(targets['edge_mean'])
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
targets['edge_mean_scaled'] = scaler.fit_transform(targets['edge_mean'].values.reshape(-1,1))

targets['edge_mean_group'] = np.where(targets['edge_mean_scaled'] > 1, "High", "Average")
targets['edge_mean_group'] = np.where(targets['edge_mean_scaled'] < -1, "Low", targets['edge_mean_group'] )

drop_idx = targets[targets['edge_mean_group'] == 'Average'].index.values

sns.boxplot(x=targets['edge_mean_group'], 
            y=targets['edge_mean_scaled'],
           palette='Set1')


# In[55]:


log = run_svm_classifier(manifold_2D_pca.drop(index=drop_idx), 
                   manifold_2D_iso.drop(index=drop_idx), 
                   manifold_2D_lle.drop(index=drop_idx), 
                   targets['edge_mean_group'].drop(index=drop_idx), 
                   targets['edge_mean_scaled'].drop(index=drop_idx), 
                   'Retrieval Network Edge Strength')

logdf = logdf.append(log, ignore_index=True)
log


# # Results

# In[56]:


logdf


# In[57]:


logdf.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/results.csv', index=None)

