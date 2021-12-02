#!/usr/bin/env python
# coding: utf-8

# # Actigraphy Preprocessing
# 
# ## Week Data

# In[1]:


import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
from matplotlib.dates import WeekdayLocator
import plotly.express as px
import seaborn as sns

from wearables import preproc


# In[2]:


actig_files = sorted(glob.glob('/Users/mcmahonmc/Box/CogNeuroLab/Aging Decision Making R01/data/actigraphy/raw/*.csv'))

# actig_files_ = random.sample(actig_files, 10)

actig_files[:5]


# In[3]:


actdf = pd.DataFrame()

weekday_map= {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu',
              4:'Fri', 5:'Sat', 6:'Sun'}

for file in actig_files:
    
    subject = file.split('raw/')[1][:5]
#     print(subject)
    
    # read in actigraphy data
    act = preproc.preproc(file, 'actiwatch', sr='.5T', truncate=False, write=False, plot=True, recording_period_min=7)
    
    # find first Monday midnight and start data from there so all subjects starting on same day of the week
    start = act[(act.index.dayofweek == 1) & (act.index.hour == 0)].index[0]
    
    # cyclically wrap days that were cut off so not losing data
    wrap = act[:start]
    wrap.index = pd.date_range(start=act[start:].last_valid_index() + pd.Timedelta(seconds=30),
                                  end=act[start:].last_valid_index() + (wrap.last_valid_index() - wrap.first_valid_index()) + pd.Timedelta(seconds=30),
                                  freq='30S')
    
    act = pd.concat((act[start:], wrap))
    
    # keep only seven days of data
    act = act[act.index <= (act.index[2] + pd.Timedelta(days=7))]
    
    x_dates = [ weekday_map[day] for day in act.index.dayofweek.unique() ]
    
    # plot
    fig, ax = plt.subplots()
    fig = sns.lineplot(x=act.index, y=act).set(title='sub-%s' % subject)
    ax.set_xticklabels(labels=x_dates, rotation=45, ha='left')
    ax.set_xlabel(''); ax.set_ylabel('Activity')
    plt.show()
    
    # if subject has < 7 days data, discard, else add to dataset
    if ( (act.last_valid_index() - act.first_valid_index()) >= pd.Timedelta(days=7) ):
    
        act = act.resample('60min').sum()
        print(act.shape)
        actdf[subject] = act.values
        print(actdf.shape)
        
        fig, ax = plt.subplots()
        fig = sns.lineplot(x=act.index, y=act, drawstyle='steps-post').set(title='sub-%s' % subject)
        ax.set_xticklabels(labels=x_dates, rotation=45, ha='left')
        ax.set_xlabel(''); ax.set_ylabel('Activity')
        plt.show()
    
    else:
        
        print('sub-%s discarded, recording period %s days' % 
              (subject, act.last_valid_index() - act.first_valid_index()))


# In[4]:


sns.heatmap(actdf.isnull(), cmap="YlGnBu")


# In[7]:


7 * 24 * 60 * 2


# In[77]:


7 * 24


# In[9]:


actdf.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/actigraphy_data_hourly_df.csv', index=False)


# ## 24 Hour Day Data

# In[2]:


actig_files = sorted(glob.glob('/Users/mcmahonmc/Box/CogNeuroLab/Aging Decision Making R01/data/actigraphy/raw/*.csv'))

# actig_files_ = random.sample(actig_files, 10)

actig_files[:5]


# In[77]:


actdf = pd.DataFrame()

weekday_map= {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu',
              4:'Fri', 5:'Sat', 6:'Sun'}

for file in actig_files:
    
    subject = file.split('raw/')[1][:5]
#     print(subject)
    
    # read in actigraphy data
    act = preproc.preproc(file, 'actiwatch', sr='.5T', truncate=False, write=True, plot=True, recording_period_min=7)
    
    # find first Monday midnight and start data from there so all subjects starting on same day of the week
    start = act[(act.index.dayofweek == 1) & (act.index.hour == 0)].index[0]
    
    # cyclically wrap days that were cut off so not losing data
    wrap = act[:start]
    wrap.index = pd.date_range(start=act[start:].last_valid_index() + pd.Timedelta(seconds=30),
                                  end=act[start:].last_valid_index() + (wrap.last_valid_index() - wrap.first_valid_index()) + pd.Timedelta(seconds=30),
                                  freq='30S')
    
    act = pd.concat((act[start:], wrap))
    
    # keep only seven days of data
    act = act[act.index <= (act.index[2] + pd.Timedelta(days=7))]
    
    x_dates = [ weekday_map[day] for day in act.index.dayofweek.unique() ]
    
    # plot
    fig, ax = plt.subplots()
    fig = sns.lineplot(x=act.index, y=act).set(title='sub-%s week' % subject)
    ax.set_xticklabels(labels=x_dates, rotation=45, ha='left')
    ax.set_xlabel('Day of Week'); ax.set_ylabel('Activity')
    plt.show()
    
    # if subject has < 7 days data, discard, else add to dataset
    if ( (act.last_valid_index() - act.first_valid_index()) >= pd.Timedelta(days=7) ):
        
        # bin into hours and take activity mean
        act = act.resample('60min').sum()
#         print(act.head())
        act24 = act.groupby(act.index.hour).mean()
#         print(act24.head())
        
        # plot
        fig, ax = plt.subplots()
        fig = sns.lineplot(x=act24.index, y=act24, 
                          drawstyle='steps-post').set(title='sub-%s 24 hr' % subject)
#         ax.set_xticklabels(labels=x_dates, rotation=45, ha='left')
        ax.set_xlabel('Hour'); ax.set_ylabel('Activity')
        plt.show()
        
        actdf[subject] = act24.values
        print(actdf.shape)
    
    else:
        
        print('sub-%s discarded, recording period %s days' % 
              (subject, act.last_valid_index() - act.first_valid_index()))


# In[8]:


actdf.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/actigraphy_data_24hrday_df.csv', index=False)


# # 24 hr non-cyclic

# In[11]:


start = act[act > 0].index[0]
act = act[start:]


# In[12]:


actdf = pd.DataFrame()

weekday_map= {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu',
              4:'Fri', 5:'Sat', 6:'Sun'}

for file in actig_files:
    
    subject = file.split('raw/')[1][:5]
#     print(subject)
    
    # read in actigraphy data
    act = preproc.preproc(file, 'actiwatch', sr='.5T', truncate=False, write=True, plot=True, recording_period_min=7)
    
    start = act[act > 0].index[0]
    act = act[start:]
    
    # keep only seven days of data
    act = act[act.index <= (act.index[2] + pd.Timedelta(days=7))]
    
    x_dates = [ weekday_map[day] for day in act.index.dayofweek.unique() ]
    
    # plot
    fig, ax = plt.subplots()
    fig = sns.lineplot(x=act.index, y=act).set(title='sub-%s week' % subject)
    ax.set_xticklabels(labels=x_dates, rotation=45, ha='left')
    ax.set_xlabel('Day of Week'); ax.set_ylabel('Activity')
    plt.show()
    
    # if subject has < 7 days data, discard, else add to dataset
    if ( (act.last_valid_index() - act.first_valid_index()) >= pd.Timedelta(days=7) ):
        
        # bin into hours and take activity mean
        act = act.resample('60min').sum()
#         print(act.head())
        act24 = act.groupby(act.index.hour).mean()
#         print(act24.head())
        
        # plot
        fig, ax = plt.subplots()
        fig = sns.lineplot(x=act24.index, y=act24, 
                          drawstyle='steps-post').set(title='sub-%s 24 hr' % subject)
#         ax.set_xticklabels(labels=x_dates, rotation=45, ha='left')
        ax.set_xlabel('Hour'); ax.set_ylabel('Activity')
        plt.show()
        
        actdf[subject] = act24.values
        print(actdf.shape)
    
    else:
        
        print('sub-%s discarded, recording period %s days' % 
              (subject, act.last_valid_index() - act.first_valid_index()))


# In[13]:


actdf.to_csv('/Users/mcmahonmc/Github/machine-learning-2021/final_project/actigraphy_data_24hrday-noncyclic_df.csv', index=False)


# In[15]:


sns.heatmap(actdf.T)


# In[ ]:




