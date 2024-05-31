"""
This script is specific to my thesis and is used to generate visualizations that will be used in the Results section.
Therefore, this script can be largely ignored.

"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter


'''
-----------------------------------------------------------------------------------------------------------------------
Extract data from the SQLite DB for visualizations 
-----------------------------------------------------------------------------------------------------------------------
'''

path = 'C:/Users/scott/PycharmProjects/SWIFr_sdm/SWIFr_HMM/hmm/output_db/'
conn = sqlite3.connect(path + 'hmm_predictions.db')
# table names that contain the stochastic backtrace, viterbi, and gamms
table_xpehh = f'sbt_prediction_xpehh_4class'
table_ihs = f'sbt_prediction_ihs_afr_std_4class'
table_fst = f'sbt_prediction_fst_4class'
swfr_table_xpehh = 'swifr_pred_xpehh_4class'
swfr_table_ihs = 'swifr_pred_ihs_4class'
swfr_table_fst = 'swifr_pred_fst_4class'
swfr_table_all = 'swifr_pred_allStats_4class'

sql_xpehh = (f"""
       SELECT *
        FROM {table_xpehh}
        WHERE vcf_name = 'ts_sweep_0.vcf'
       """)

sql_ihs = (f"""
       SELECT *
        FROM {table_ihs}
       """)

sql_fst = (f"""
       SELECT *
        FROM {table_fst}
       """)

swfr_sql_xpehh = (f"""
       SELECT *
        FROM {swfr_table_xpehh}
        WHERE vcf_name = 'ts_sweep_1.vcf'
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
# ihs = pd.read_sql(sql_ihs, conn)
# fst = pd.read_sql(sql_fst, conn)
#
# # drop nans prior to analysis
xpehh = xpehh[xpehh['xpehh'] != -998.0].reset_index(drop=True)
xpehh_classes = list(xpehh['label'].unique())
# ihs = ihs[ihs['ihs_afr_std'] != -998.0].reset_index(drop=True)
# ihs_classes = list(ihs['label'].unique())
# fst = fst[fst['fst'] != -998.0].reset_index(drop=True)
# fst_classes = list(fst['label'].unique())
# # swifr data
sxpehh = pd.read_sql(swfr_sql_xpehh, conn)
# sihs = pd.read_sql(swfr_sql_ihs, conn)
# sfst = pd.read_sql(swfr_sql_fst, conn)
# sall = pd.read_sql(swfr_sql_all, conn)
# # drop null values
sxpehh = sxpehh[sxpehh['xpehh'] != -998.0].reset_index(drop=True)
sxpehh_classes = list(sxpehh['label'].unique())
# sihs = sihs[sihs['ihs_afr_std'] != -998.0].reset_index(drop=True)
# sihs_classes = list(sihs['label'].unique())
# sfst = sfst[sfst['fst'] != -998.0].reset_index(drop=True)
# sfst_classes = list(sfst['label'].unique())
# sall = sall[(sall['fst'] != -998.0) |
#             (sall['xpehh'] != -998.0) |
#             (sall['ihs_afr_std'] != -998.0)].reset_index(drop=True)
# sall_classes = list(sall['label'].unique())
# # add class labels for each row
swfr_cols = ['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)']
sxpehh['label_pred'] = sxpehh[swfr_cols].idxmax(axis=1)
sxpehh['label_pred'] = sxpehh['label_pred'].replace(['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)'],
                                      ['neutral', 'link_left', 'link_right', 'sweep'])
# sihs['label_pred'] = sihs[swfr_cols].idxmax(axis=1)
# sihs['label_pred'] = sihs['label_pred'].replace(['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)'],
#                                       ['neutral', 'link_left', 'link_right', 'sweep'])
# sfst['label_pred'] = sfst[swfr_cols].idxmax(axis=1)
# sfst['label_pred'] = sfst['label_pred'].replace(['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)'],
#                                       ['neutral', 'link_left', 'link_right', 'sweep'])
# sall['label_pred'] = sall[swfr_cols].idxmax(axis=1)
# sall['label_pred'] = sall['label_pred'].replace(['P(neutral)', 'P(link_left)', 'P(link_right)', 'P(sweep)'],
#                                       ['neutral', 'link_left', 'link_right', 'sweep'])

""" 
---------------------------------------------------------------------------------------------------
Plot -- Path and stat comparison [flashlight plot]
---------------------------------------------------------------------------------------------------
"""
state2num = {'label_actual': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
state2numPred = {'viterbi_class_xpehh_num': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}

xpehh['label_actual'] = xpehh['label']
xpehh['viterbi_class_xpehh_num'] = xpehh['viterbi_class_xpehh']
xpehh = xpehh.replace(state2num)
xpehh = xpehh.replace(state2numPred)

xs = np.arange(0, len(xpehh), 1)
xs2 = np.arange(0, len(sxpehh), 1)
cmap = mpl.colormaps['cool']
legend_colors = cmap(np.linspace(0, 1, len(xpehh_classes)))
colors = ['magenta', 'dodgerblue', 'darkviolet', 'blue']  # enough for four classes

fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(5, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle(f'Actual Path, Predicted Path, and XP-EHH')
# axs[0].plot(xs, xpehh['label'], color='black')
axs[0].scatter(xs, [0]*len(xs), c=xpehh['label_actual'], cmap='cool', marker=",", s=10)
axs[0].axvline(x=xpehh[xpehh['label']=='sweep'].index, color='black', label='Sweep Position')
# axs[0].scatter(xs, xpehh['label'], c=colors, cmap='viridis', edgecolor='none', s=30)
# axs[0].scatter(xs, xpehh['label'], s=30)

# axs[1].plot(xs, xpehh['viterbi_class_xpehh'], color='black')
axs[1].scatter(xs, [1]*len(xs), c=xpehh['viterbi_class_xpehh_num'], cmap='cool', marker=",", s=10)
axs[1].axvline(x=xpehh[xpehh['label']=='sweep'].index, color='black', label='Sweep Position')
# axs[1].scatter(xs, xpehh['viterbi_class_xpehh'], c=colors, cmap='viridis', edgecolor='none', s=30)
# axs[1].scatter(xs, xpehh['viterbi_class_xpehh'],  s=30)
for i in range(100):
    axs[2].plot(xs, xpehh.loc[:, f'sb_{i}'], color='lightgrey')
# axs[2].scatter(xs, sb_paths, c=sb_paths,cmap='viridis', edgecolor='none', s=30)

axs[3].plot(xs2, sxpehh['label_pred'], color='black')
# axs[3].scatter(xs2, sxpehh['label_pred'], c=colors, cmap='viridis', edgecolor='none', s=30)
axs[3].scatter(xs2, sxpehh['label_pred'], s=30)

# axs[4].scatter(xs, xpehh['xpehh'], c=xpehh['xpehh'], cmap='viridis', edgecolor='none', s=3)
axs[4].scatter(xs, xpehh['xpehh'], c=xpehh['xpehh'], s=3)

axs[0].set(ylabel='Actual State')
axs[1].set(ylabel='Viterbi Pred')
axs[2].set(ylabel='BackTrc Pred')
axs[3].set(ylabel='SWIFr Pred')
axs[4].set(ylabel='Value XP-EHH')
# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

legend_elements = [Line2D([0], [0], marker='o', color='w', label='0: Neutral',
                          markerfacecolor=legend_colors[0], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='1: Link_Left',
                          markerfacecolor=legend_colors[1], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='2: Link_Right',
                          markerfacecolor=legend_colors[2], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='3: Sweep',
                          markerfacecolor=legend_colors[3], markersize=15)]

axs[0].legend(handles=legend_elements, loc='upper left')
# axs[1].legend(handles=legend_elements, loc='upper left')
plt.show()





