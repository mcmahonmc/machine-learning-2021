neuro = pd.read_csv('/Users/mcmahonmc/Box/CogNeuroLab/Aging Decision Making R01/data/neuropsych/AgingDecMemNeuropsyc_DATA_2021-11-17_0900.csv')

neuro = neuro.dropna(subset=['trails_b_z_score'])

neuro['trails_b_group'] = np.where(neuro['trails_b_z_score'] < np.percentile(neuro['trails_b_z_score'], 33),
                                  'Low', 'Medium')
neuro['trails_b_group'] = np.where(neuro['trails_b_z_score'] > np.percentile(neuro['trails_b_z_score'], 66),
                                  'High', neuro['trails_b_group'])

drop_subs = [ int(subject) for subject in neuro['participant_id'] if str(subject) not in actdf.columns.values ]
drop_subs

neuro = (neuro[~neuro['participant_id'].isin(drop_subs)])
neuro = neuro.reset_index()
neuro.index = list(neuro.index)

actdf = actdf.drop([ str(subject) for subject in actdf.columns.values if int(subject) not in neuro['participant_id'].values ], axis=1)

neuro[['participant_id', 'age', 'gender', 'trails_b_z_score', 'trails_b_group']]

##

brain = pd.read_csv('/Volumes/schnyer/Megan/adm_mem-fc/data/dataset_2021-11-10.csv')
brain = brain.drop(['IS', 'IV', 'RA', 'L5', 'L5_starttime', 'M10', 'M10_starttime',
                   'actamp', 'actmin', 'actbeta', 'actphi', 'actalph'], axis=1)

drop_subs = [ int(subject) for subject in brain['subject'] if str(subject) not in actdf.columns.values ]

brain = (brain[~brain['subject'].isin(drop_subs)])
brain = brain.reset_index()

actdf = actdf.drop([ str(subject) for subject in actdf.columns.values if int(subject) not in brain['subject'].values ], axis=1)

brain.shape

##

neuro[['participant_id', 'age', 'gender', 'trails_b_z_score', 'trails_b_group']].merge(brain, left_on = 'participant_id', right_on = 'subject').drop('participant_id', axis=1).set_index('subject').to_csv(
'/Users/mcmahonmc/Github/machine-learning-2021/final/target_data.csv', index=True)
