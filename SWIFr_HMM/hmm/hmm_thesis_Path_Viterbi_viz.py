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
        WHERE vcf_name IN ('ts_sweep_10.vcf')
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
        WHERE vcf_name IN ('ts_sweep_10.vcf')
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
# xs = np.arange(0, len(xpehh), 1)
# xs2 = np.arange(0, len(sxpehh), 1)

xs = xpehh['snp_position']
xs2 = sxpehh['snp_position']
cmap = mpl.colormaps['cool']
legend_colors = cmap(np.linspace(0, 1, len(xpehh_classes)))

fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(5, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle(f'Actual Path, Predicted Path, and XP-EHH')

state2numColor = {'label_actual_color': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
state2numPlot = {'label_actual_plot': {'neutral': 0, 'link_left': 1, 'link_right': 1, 'sweep': 2}}
xpehh['label_actual_color'] = xpehh['label']
xpehh['label_actual_plot'] = xpehh['label']
xpehh = xpehh.replace(state2numColor)
xpehh = xpehh.replace(state2numPlot)
axs[0].axvline(x=2.5e6, color='black')
axs[0].plot(xs, xpehh['label_actual_plot'], color='lightgrey', alpha=0.5)
axs[0].scatter(xs, xpehh['label_actual_plot'], c=xpehh['label_actual_color'],
               cmap='cool', edgecolor='none', s=30, alpha=0.7)

state2numColor = {'viterbi_class_xpehh_plot': {'neutral': 0, 'link_left': 1, 'link_right': 1, 'sweep': 2}}
state2numPlot = {'viterbi_class_xpehh_color': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
xpehh['viterbi_class_xpehh_plot'] = xpehh['viterbi_class_xpehh']
xpehh['viterbi_class_xpehh_color'] = xpehh['viterbi_class_xpehh']
xpehh = xpehh.replace(state2numColor)
xpehh = xpehh.replace(state2numPlot)
axs[1].axvline(x=2.5e6, color='black')
axs[1].plot(xs, xpehh['viterbi_class_xpehh_plot'], color='lightgrey', alpha=0.5)
axs[1].scatter(xs, xpehh['viterbi_class_xpehh_plot'], c=xpehh['viterbi_class_xpehh_color'],
               cmap='cool', edgecolor='none', s=30, alpha=0.7)

for i in range(100):
    state2numColor = {f'sb_{i}_plot': {'neutral': 0, 'link_left': 1, 'link_right': 1, 'sweep': 2}}
    state2numPlot = {f'sb_{i}_color': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
    xpehh[f'sb_{i}_plot'] = xpehh[f'sb_{i}']
    xpehh[f'sb_{i}_color'] = xpehh[f'sb_{i}']
    xpehh = xpehh.replace(state2numColor)
    xpehh = xpehh.replace(state2numPlot)
    axs[2].axvline(x=2.5e6, color='black')
    axs[2].plot(xs, xpehh[f'sb_{i}_plot'], color='lightgrey', alpha=0.1)
    axs[2].scatter(xs, xpehh[f'sb_{i}_plot'], c=xpehh[f'sb_{i}_color'], cmap='cool', edgecolor='none', s=30, alpha=0.7)

maxval = (len(sxpehh['label_pred'].unique()))/(len(sxpehh_classes))
new_cmap = truncate_colormap(plt.get_cmap('cool'), minval=0, maxval=maxval, n=100)
state2numColor = {'label_pred_color': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
state2numPlot = {'label_pred_plot': {'neutral': 0, 'link_left': 1, 'link_right': 1, 'sweep': 2}}
sxpehh['label_pred_color'] = sxpehh['label_pred']
sxpehh['label_pred_plot'] = sxpehh['label_pred']
sxpehh = sxpehh.replace(state2numColor)
sxpehh = sxpehh.replace(state2numPlot)
axs[3].axvline(x=2.5e6, color='black')
axs[3].plot(xs2, sxpehh['label_pred_plot'], color='lightgrey', alpha=0.5)
axs[3].scatter(xs2, sxpehh['label_pred_plot'], c=sxpehh['label_pred_color'], cmap=new_cmap, edgecolor='none', s=30)

axs[4].axvline(x=2.5e6, color='black')
stats_plot = axs[4].scatter(xs, xpehh['xpehh'], c=xpehh['xpehh'], cmap='cool', s=3)

# Add colorbars inside the stats plot (bottom plot)
cax1 = inset_axes(axs[4], width="2%", height="100%", loc='right', borderpad=0)
cbar4 = fig.colorbar(stats_plot, cax=cax1)

axis_labels = ['', 'Neutral', 'Link (left/right)','Sweep', '']
axis_ticks = [-0.5, 0, 1, 2, 2.5]
axs[0].set(ylabel='Actual State')
axs[0].yaxis.set_ticks(axis_ticks)
axs[0].set_yticklabels(axis_labels)

axs[1].set(ylabel='Viterbi Pred')
axs[1].yaxis.set_ticks(axis_ticks)
axs[1].set_yticklabels(axis_labels)

axs[2].set(ylabel='BackTrc Pred')
axs[2].yaxis.set_ticks(axis_ticks)
axs[2].set_yticklabels(axis_labels)

axs[3].set(ylabel='SWIFr Pred')
axs[3].yaxis.set_ticks(axis_ticks)
axs[3].set_yticklabels(axis_labels)

axs[4].set(ylabel='Value XP-EHH')
# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Neutral',
                          markerfacecolor=legend_colors[0], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Link_Left',
                          markerfacecolor=legend_colors[1], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Link_Right',
                          markerfacecolor=legend_colors[2], markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Sweep',
                          markerfacecolor=legend_colors[3], markersize=15),
                   Line2D([0], [0], marker='_', color='black', label='Sweep Actual',
                          markersize=10)
                   ]

axs[0].legend(handles=legend_elements, loc='upper left')
plt.show()





