"""
This script is specific to my thesis and is used to generate visualizations that will be used in the Results section.
Therefore, this script can be largely ignored.

"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

'''
-----------------------------------------------------------------------------------------------------------------------
Extract data from the SQLite DB for visualizations 
-----------------------------------------------------------------------------------------------------------------------
'''
states = 4  # change from 2, 3, or 4 states
path = 'C:/Users/scott/PycharmProjects/SWIFr_sdm/SWIFr_HMM/hmm/output_db/'
# path = 'C:/Users/scott/PycharmProjects/population_simulations/db_build/'
# conn = sqlite3.connect(path + 'population_simulation_v3.db')
conn = sqlite3.connect(path + 'hmm_predictions.db')
# table names that contain the stochastic backtrace, viterbi, and gamms
table = f'hmm_predictions_{states}class'
# table = 'sweep_simulations_stats'

sql = (f"""
       SELECT *
        FROM {table}
       """)

# collect a list of the unique simulations
df = pd.read_sql(sql, conn)
classes = list(df['label'].unique())
# drop nans prior to analysis
xpehh = df[df['xpehh'] != -998.0].reset_index(drop=True)
ihs = df[df['ihs_afr_std'] != -998.0].reset_index(drop=True)
fst = df[df['fst'] != -998.0].reset_index(drop=True)

'''
-----------------------------------------------------------------------------------------------------------------------
HISTOGRAM - XP-EHH Distribution
-----------------------------------------------------------------------------------------------------------------------
'''
n_bins = 100
data_list = [xpehh, fst, ihs]
stats = ['xpehh', 'fst', 'ihs_afr_std']
titles = ['XP-EHH', 'Fst', 'iHS']

colors = ['magenta', 'dodgerblue', 'darkviolet', 'blue']

for i, data in enumerate(data_list):
       plt.figure(figsize=(12,5))
       # plt.hist(data[data['label']=='neutral'][stats[i]], density=True, label='neutral', bins=n_bins+500, alpha=0.1, color=colors[0])
       sns.kdeplot(data=data[data['label'] == 'neutral'], x=stats[i], color=colors[0], fill=True)

       # plt.hist(data[data['label']=='link_left'][stats[i]], density=True, label='link_left', bins=n_bins+300, alpha=0.1, color=colors[1])
       sns.kdeplot(data=data[data['label'] == 'link_left'], x=stats[i], color=colors[1], fill=True)

       # plt.hist(data[data['label']=='link_right'][stats[i]], density=True, label='link_right', bins=n_bins+300, alpha=0.1, color=colors[2])
       sns.kdeplot(data=data[data['label'] == 'link_right'], x=stats[i], color=colors[2], fill=True)

       # plt.hist(data[data['label']=='sweep'][stats[i]], density=True, label='sweep', bins=10, alpha=0.1, color=colors[3])
       sns.kdeplot(data=data[data['label'] == 'sweep'], x=stats[i], color=colors[3], fill=True)

       plt.legend(labels=['neutral', 'link_left', 'link_right', 'sweep'])
       plt.title(f'{titles[i]} Distribution')
       plt.savefig(f'plots_thesis/distribution_{stats[i]}.svg')
       plt.show()


# plt.figure(figsize=(12, 5))
# plt.hist(data[data['label']=='neutral'][stat], density=True, label='neutral', bins=n_bins+500, alpha=0.7, color='green')
# plt.hist(data[data['label']=='link_left'][stat], density=True, label='link_left', bins=n_bins+300, alpha=0.7, color='red')
# plt.hist(data[data['label']=='link_right'][stat], density=True, label='link_right', bins=n_bins+300, alpha=0.7, color='tomato')
# plt.hist(data[data['label']=='sweep'][stat], density=True, label='sweep', bins=n_bins, alpha=0.7, color='blue')
# plt.xlabel('XP-EHH Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of XP-EHH')
# plt.legend()
# plt.show()
# print('done')






