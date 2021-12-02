#!/usr/bin/env python
# coding: utf-8

# # Standard rest-activity measure calculation

# In[1]:


import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
from matplotlib.dates import WeekdayLocator
import seaborn as sns


# In[2]:


actdf = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/actigraphy_data_hourly_df.csv')
print('actigraphy df')
print(actdf.shape)


# In[3]:


from wearables import fitcosinor, npmetrics
from datetime import datetime

rar = pd.DataFrame()

for subject in actdf.columns:
    
    df = pd.DataFrame(actdf[subject][:-2]).set_index(pd.to_datetime(
        pd.date_range(start = pd.to_datetime('2021-01-01 00:00:00'),
                      end = pd.to_datetime('2021-01-01 00:00:00') + pd.Timedelta(days=7),
                      freq='30S'),
        format = '%Y-%m-%d %H:%M:%S'))
    
    df.columns = ['Activity']
    
    cr = np.array(fitcosinor.fitcosinor(df)[0].T.values).T[0]
    nonp = npmetrics.np_metrics_all(df['Activity'])
    
    rar[subject] = np.concatenate((cr, nonp[:3]))
    
rar = rar.T
rar.columns = ['actmin', 'amp', 'alpha', 'beta', 'phi', 'IS', 'IV', 'RA']
rar


# In[13]:


rar = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/rar_df.csv', index_col=0)

drop_subs = [ int(subject) for subject in rar.index if str(subject) not in actdf.columns.values ]
drop_subs

rar2 = (rar[~rar.index.isin(drop_subs)])
print(rar2.shape)


# In[14]:


[col for col in targets.columns if 'mean_active' in col]


# In[15]:


[col for col in targets.columns if 'mean_sleep' in col]


# In[16]:


rar2 = rar2.merge(targets[['total_ac_mean_active',
                    'duration_mean_sleep', 
                    'total_ac_mean_sleep', 
                   'efficiency_mean_sleep',
                   'sleep_time_mean_sleep',
                   'sleep_time_sd_sleep',
                   'onset_latency_mean_sleep']],
          left_index=True, right_index=True)

sns.heatmap(rar2.isnull(), cmap='terrain')


# In[17]:


# rar.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/rar_df.csv')


# In[121]:


rar = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/rar_df.csv')

targets = pd.read_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/target_data.csv')
targets['edge_mean'] = targets[[x for x in targets.columns if x.startswith('edge_')]].mean(axis=1)


# In[122]:


pd.merge(rar, targets, left_on = 'Unnamed: 0', right_on = 'subject')


# In[16]:


[col for col in targets.columns if 'mean_active' in col]


# In[126]:


corr = pd.merge(rar, targets[[col for col in targets.columns if 'z_score' in col] +
                                   [col for col in targets.columns if 'zscore' in col] + 
                             ['subject']],
                left_on = 'Unnamed: 0', right_on = 'subject').drop(['subject', 'Unnamed: 0'], axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr[np.abs(corr) > 0.2], mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:




