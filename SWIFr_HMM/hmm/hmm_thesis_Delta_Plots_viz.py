"""
This script is specific to my thesis and is used to generate visualizations that will be used in the Results section.
Therefore, this script can be largely ignored.

"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as matcolors
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # compliments of stack:
    # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = matcolors.LinearSegmentedColormap.from_list(
        f'{cmap.name}, {minval}, {maxval}',
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

'''
-----------------------------------------------------------------------------------------------------------------------
Extract data from the SQLite DB for visualizations 
-----------------------------------------------------------------------------------------------------------------------
'''
save_fig = 'yes'
plot_save = 'delta_plot_fst_4class_allSims.png'

path = 'C:/Users/scott/PycharmProjects/SWIFr_sdm/SWIFr_HMM/hmm/output_db/'
conn = sqlite3.connect(path + 'hmm_predictions.db')
# table names that contain the stochastic backtrace, viterbi, and gamma
table_xpehh = 'sbt_prediction_xpehh_4class'
table_ihs = 'sbt_prediction_ihs_afr_std_4class'
table_fst = 'sbt_prediction_fst_4class'
swfr_table_xpehh = 'swifr_pred_xpehh_4class'
swfr_table_ihs = 'swifr_pred_ihs_4class'
swfr_table_fst = 'swifr_pred_fst_4class'
swfr_table_all = 'swifr_pred_allStats_4class'

sql_xpehh = (f"""
       SELECT *
        FROM {table_xpehh}
        WHERE xpehh != -998.0
       """)

sql_ihs = (f"""
       SELECT *
        FROM {table_ihs}
        WHERE ihs_afr_std != -998.0
       """)

sql_fst = (f"""
       SELECT *
        FROM {table_fst}
        WHERE fst != -998.0
       """)

swfr_sql_xpehh = (f"""
       SELECT *
        FROM {swfr_table_xpehh}
       """)

swfr_sql_ihs = (f"""
       SELECT *
        FROM {swfr_table_ihs}
       """)

swfr_sql_fst = (f"""
       SELECT *
        FROM {swfr_table_fst}
       """)

swfr_sql_all = (f"""
       SELECT *
        FROM {swfr_table_all}
       """)

# collect a list of the unique simulations

xpehh = pd.read_sql(sql_xpehh, conn)
ihs = pd.read_sql(sql_ihs, conn)
fst = pd.read_sql(sql_fst, conn)

# drop nans prior to analysis
# xpehh = xpehh[xpehh['xpehh'] != -998.0].reset_index(drop=True)
# ihs = ihs[ihs['ihs_afr_std'] != -998.0].reset_index(drop=True)
# fst = fst[fst['fst'] != -998.0].reset_index(drop=True)

# # swifr data
sxpehh = pd.read_sql(swfr_sql_xpehh, conn)
sihs = pd.read_sql(swfr_sql_ihs, conn)
sfst = pd.read_sql(swfr_sql_fst, conn)
sall = pd.read_sql(swfr_sql_all, conn)
#
# # drop null values
sxpehh = sxpehh[sxpehh['xpehh'] != -998.0].reset_index(drop=True)
sihs = sihs[sihs['ihs_afr_std'] != -998.0].reset_index(drop=True)
sfst = sfst[sfst['fst'] != -998.0].reset_index(drop=True)
sall = sall[(sall['fst'] != -998.0) |
            (sall['xpehh'] != -998.0) |
            (sall['ihs_afr_std'] != -998.0)].reset_index(drop=True)
#
# # add class labels for each row
swfr_cols = ['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)']
sxpehh['label_pred'] = sxpehh[swfr_cols].idxmax(axis=1)
sxpehh['label_pred'] = sxpehh['label_pred'].replace(['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)'],
                                      ['neutral', 'link_left', 'link_right', 'sweep'])
sihs['label_pred'] = sihs[swfr_cols].idxmax(axis=1)
sihs['label_pred'] = sihs['label_pred'].replace(['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)'],
                                      ['neutral', 'link_left', 'link_right', 'sweep'])
sfst['label_pred'] = sfst[swfr_cols].idxmax(axis=1)
sfst['label_pred'] = sfst['label_pred'].replace(['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)'],
                                      ['neutral', 'link_left', 'link_right', 'sweep'])
sall['label_pred'] = sall[swfr_cols].idxmax(axis=1)
sall['label_pred'] = sall['label_pred'].replace(['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)'],
                                      ['neutral', 'link_left', 'link_right', 'sweep'])

# define df that contains all the sub frames
frames = [ihs, xpehh, fst, sxpehh, sfst, sihs, sall]
print('calculating delta sweep')
for f in frames:
    # the hard sweep is known and fixed at snp position 2.5e6
    f['delta_sweep'] = f['snp_position'] - 2.5e6

''' flatten the stochastic back traces '''
# define sbt columns
col_list = ihs.columns # any stat could be used, columns are the same for all
sb_list = [i for i in col_list if 'sb_' in i]
# define empty frames to hold flattened data
sb_ihs = pd.DataFrame()
sb_xpehh = pd.DataFrame()
sb_fst = pd.DataFrame()
# might be able to accelerate this with np.flatten, but need both the sb columns and delta_sweep column
print('Flattening stochastic backtraces...')
for sb in sb_list:
    ihs2 = ihs[ihs[sb] == 'sweep'][sb].tolist()
    ihs3 = ihs[ihs[sb] == 'sweep']['delta_sweep'].tolist()
    ihs4 = pd.DataFrame({'sb_stack':ihs2, 'delta_sweep':ihs3})
    sb_ihs = pd.concat([sb_ihs, ihs4], ignore_index=True)

    xpehh2 = xpehh[xpehh[sb] == 'sweep'][sb].tolist()
    xpehh3 = xpehh[xpehh[sb] == 'sweep']['delta_sweep'].tolist()
    xpehh4 = pd.DataFrame({'sb_stack':xpehh2, 'delta_sweep':xpehh3})
    sb_xpehh = pd.concat([sb_xpehh, xpehh4], ignore_index=True)

    fst2 = fst[fst[sb] == 'sweep'][sb].tolist()
    fst3 = fst[fst[sb] == 'sweep']['delta_sweep'].tolist()
    fst4 = pd.DataFrame({'sb_stack':fst2, 'delta_sweep':fst3})
    sb_fst = pd.concat([sb_fst, fst4], ignore_index=True)


"""
------------------------------------------------------------------------------------------------------------------------
Delta Distribution Plots
------------------------------------------------------------------------------------------------------------------------
"""
''' ---- XP-EHH --- '''
colors = ['magenta', 'dodgerblue', 'darkviolet', 'blue']
plt.figure(figsize=(12, 5))
sns.kdeplot(data=sb_xpehh, x='delta_sweep', color=colors[0], fill=True)
sns.kdeplot(data=xpehh[xpehh['viterbi_class_xpehh'] == 'sweep'], x='delta_sweep', color=colors[1], fill=True)
if len(sxpehh[sxpehh['label_pred'] == 'sweep']) > 0:
    sns.kdeplot(data=sxpehh[sxpehh['label_pred'] == 'sweep'], x='delta_sweep', color=colors[2], fill=True)
sns.kdeplot(data=sall[sall['label_pred'] == 'sweep'], x='delta_sweep', color=colors[3], fill=True)
if len(sxpehh[sxpehh['label_pred'] == 'sweep']) > 0:
    plt.legend(labels=['Stochastic Backtrace', 'Viterbi Path', 'SWIF(r) 1-Stat', 'SWIF(r) All-Stats'])
else:
    plt.legend(labels=['Stochastic Backtrace', 'Viterbi Path', 'SWIF(r) All-Stats'])

plt.title(f'XP-EHH Delta Distribution from Sweep Actual')
plt.savefig(f'plots_thesis/delta_distribution_xpehh.svg')
plt.show()

''' ---- iHS --- '''
colors = ['magenta', 'dodgerblue', 'darkviolet', 'blue']
plt.figure(figsize=(12, 5))
sns.kdeplot(data=sb_ihs, x='delta_sweep', color=colors[0], fill=True)
sns.kdeplot(data=ihs[ihs['viterbi_class_ihs_afr_std'] == 'sweep'], x='delta_sweep', color=colors[1], fill=True)
if len(sihs[sihs['label_pred'] == 'sweep']) > 0:
    sns.kdeplot(data=sihs[sihs['label_pred'] == 'sweep'], x='delta_sweep', color=colors[2], fill=True)
sns.kdeplot(data=sall[sall['label_pred'] == 'sweep'], x='delta_sweep', color=colors[3], fill=True)
if len(sihs[sihs['label_pred'] == 'sweep']) > 0:
    plt.legend(labels=['Stochastic Backtrace', 'Viterbi Path', 'SWIF(r) 1-Stat', 'SWIF(r) All-Stats'])
else:
    plt.legend(labels=['Stochastic Backtrace', 'Viterbi Path', 'SWIF(r) All-Stats'])

plt.title(f'iHS Delta Distribution from Sweep Actual')
plt.savefig(f'plots_thesis/delta_distribution_ihs.svg')
plt.show()

''' ---- Fst --- '''
plt.figure(figsize=(12, 5))
sns.kdeplot(data=sb_fst, x='delta_sweep', color=colors[0], fill=True)
sns.kdeplot(data=fst[fst['viterbi_class_fst'] == 'sweep'], x='delta_sweep', color=colors[1], fill=True)
if len(sfst[sfst['label_pred'] == 'sweep']) > 0:
    sns.kdeplot(data=sfst[sfst['label_pred'] == 'sweep'], x='delta_sweep', color=colors[2], fill=True)
sns.kdeplot(data=sall[sall['label_pred'] == 'sweep'], x='delta_sweep', color=colors[3], fill=True)
if len(sfst[sfst['label_pred'] == 'sweep']) > 0:
    plt.legend(labels=['Stochastic Backtrace', 'Viterbi Path', 'SWIF(r) 1-Stat', 'SWIF(r) All-Stats'])
else:
    plt.legend(labels=['Stochastic Backtrace', 'Viterbi Path', 'SWIF(r) All-Stats'])

plt.title(f'Fst Delta Distribution from Sweep Actual')
plt.savefig(f'plots_thesis/delta_distribution_fst.svg')
plt.show()
