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
# path = 'C:/Users/scott/PycharmProjects/SWIFr_sdm/SWIFr_HMM/hmm/output_db/'
# conn = sqlite3.connect(path + 'hmm_predictions.db')
# table = f'hmm_predictions_{states}class'

path = 'C:/Users/scott/PycharmProjects/population_simulations/db_build/'
conn = sqlite3.connect(path + 'population_simulation_v3.db')
table = 'sweep_simulations_stats'
stats = ['xpehh', 'fst', 'ihs_afr_std']
stats_plot = ['XP-EHH', 'Fst', 'iHS']
stat_rename = {'xpehh':'XP-EHH', 'fst':'Fst', 'ihs_afr_std':'iHS'}

sql_sweep = (f"""
       SELECT *
        FROM {table}
        WHERE Label = 'sweep'
       """)

sql_link = (f"""
       SELECT *
        FROM {table}
        WHERE Label IN ('link_left', 'link_right')
       """)

sql_neutral = (f"""
       SELECT *
        FROM {table}
        WHERE Label = 'neutral'
       """)

'''
Sweep Data
'''
# collect a list of the unique simulations
s_all = pd.read_sql(sql_sweep, conn)
s_all = s_all.rename(columns=stat_rename)
# drop nans prior to analysis (performed for sweep, but should not have any null sweep stats)
s = s_all.dropna(subset=stats_plot, how='any').reset_index(drop=True)

# correlation matrix for sweeps
s_corr = s[stats_plot].corr()
count = len(s)
''' Sweep - Correlation Matrix Plot '''
ax = sns.heatmap(s_corr, annot=True, cmap='cool')
plt.title(f'Correlation Matrix - Sweep Statistics (n={count})')
plt.savefig(f'plots_thesis/corr_matrix_sweep.svg')
plt.show()

print("Length raw sweep data: ", len(s_all))
print("Length sweep data, no nulls: ", len(s), '\n')

'''
link Data
'''
# collect a list of the unique simulations
l_all = pd.read_sql(sql_link, conn)
l_all = l_all.rename(columns=stat_rename)
# drop nans prior to analysis
l = l_all.dropna(subset=stats_plot, how='any').reset_index(drop=True)

# correlation matrix for links (link_left and link_right)
l_corr = l[stats_plot].corr()
count = len(l)
''' Sweep - Correlation Matrix Plot '''
ax = sns.heatmap(l_corr, annot=True, cmap='cool')
plt.title(f'Correlation Matrix - Link Statistics (n={count})')
plt.savefig(f'plots_thesis/corr_matrix_link.svg')
plt.show()

print("Length raw link data: ", len(l_all))
print("Length link data, no nulls: ", len(l), '\n')

'''
neutral Data
'''
# collect a list of the unique simulations
n_all = pd.read_sql(sql_neutral, conn)
n_all = n_all.rename(columns=stat_rename)
# drop nans prior to analysis
n = n_all.dropna(subset=stats_plot, how='any').reset_index(drop=True)

# correlation matrix for sweeps
n_corr = n[stats_plot].corr()
count = len(n)
''' Sweep - Correlation Matrix Plot '''
ax = sns.heatmap(n_corr, annot=True, cmap='cool')
plt.title(f'Correlation Matrix - Neutral Statistics (n={count})')
plt.savefig(f'plots_thesis/corr_matrix_neutral.svg')
plt.show()

print("Length raw neutral data: ", len(n_all))
print("Length neutral data, no nulls: ", len(n), '\n')