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
statistic = 'fst'  # xpehh, fst, ihs_afr_std
sim_num = 19 # any number from 0 to 19

sql_xpehh = (f"""
       SELECT *
        FROM {table_xpehh}
        WHERE vcf_name IN ('ts_sweep_{sim_num}.vcf')
       """)

sql_ihs = (f"""
       SELECT *
        FROM {table_ihs}
        WHERE vcf_name IN ('ts_sweep_{sim_num}.vcf')
       """)

sql_fst = (f"""
       SELECT *
        FROM {table_fst}
        WHERE vcf_name IN ('ts_sweep_{sim_num}.vcf')
       """)

swfr_sql_xpehh = (f"""
       SELECT *
        FROM {swfr_table_xpehh}
        WHERE vcf_name IN ('ts_sweep_{sim_num}.vcf')
       """)

swfr_sql_ihs = (f"""
       SELECT *
        FROM {swfr_table_ihs}
        WHERE vcf_name IN ('ts_sweep_{sim_num}.vcf')
       """)

swfr_sql_fst = (f"""
       SELECT *
        FROM {swfr_table_fst}
        WHERE vcf_name IN ('ts_sweep_{sim_num}.vcf')
       """)

swfr_sql_all = (f"""
       SELECT *
        FROM {swfr_table_all}
        WHERE vcf_name IN ('ts_sweep_{sim_num}.vcf')
       """)

# collect a list of the unique simulations

hmm_stat = pd.read_sql(sql_xpehh, conn)
# allow for swapping between stats
if statistic == 'xpehh':
    hmm_stat = pd.read_sql(sql_xpehh, conn)
elif statistic =='ihs_afr_std':
    hmm_stat = pd.read_sql(sql_ihs, conn)
elif statistic == 'fst':
    hmm_stat = pd.read_sql(sql_fst, conn)

# drop nans prior to analysis and define list of classes
hmm_stat = hmm_stat[hmm_stat[statistic] != -998.0].reset_index(drop=True)
actual_classes = list(hmm_stat['label'].unique())

# swifr data
sxpehh = pd.read_sql(swfr_sql_xpehh, conn)
sihs = pd.read_sql(swfr_sql_ihs, conn)
sfst = pd.read_sql(swfr_sql_fst, conn)
sall = pd.read_sql(swfr_sql_all, conn)

# drop null values
sxpehh = sxpehh[sxpehh['xpehh'] != -998.0].reset_index(drop=True)
sxpehh_classes = list(sxpehh['label'].unique())
sihs = sihs[sihs['ihs_afr_std'] != -998.0].reset_index(drop=True)
sihs_classes = list(sihs['label'].unique())
sfst = sfst[sfst['fst'] != -998.0].reset_index(drop=True)
sfst_classes = list(sfst['label'].unique())

# all swifer data
sall = sall[(sall['fst'] != -998.0) |
            (sall['xpehh'] != -998.0) |
            (sall['ihs_afr_std'] != -998.0)].reset_index(drop=True)
sall_classes = list(sall['label'].unique())
# add class labels for each row
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

# allow for swapping between stats
if statistic == 'xpehh':
    swifr_select = sxpehh
    swifr_select_classes = sxpehh_classes
elif statistic =='ihs_afr_std':
    swifr_select = sihs
    swifr_select_classes = sihs_classes
elif statistic == 'fst':
    swifr_select = sfst
    swifr_select_classes = sfst_classes

""" 
---------------------------------------------------------------------------------------------------
Path Plot
---------------------------------------------------------------------------------------------------
"""
# size of markers for scatter plots
size = 50
# define x axes for brevity
xs = hmm_stat['snp_position']
xs2 = swifr_select['snp_position']
xs3 = sall['snp_position']
# define colormap approved by addy
cmap = mpl.colormaps['cool']
# define legend
legend_colors = cmap(np.linspace(0, 1, len(actual_classes)))
# initialize the figure space
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(6, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle('XP-EHH: Actual Path, Viterbi Path, Stochastic Backtrace, and XP-EHH Statistic')

''' Actual Path '''
# relabeling strings as numbers for plotting purposes
state2numColor = {'label_actual_color': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
state2numPlot = {'label_actual_plot': {'neutral': 0, 'link_left': 1, 'link_right': 1, 'sweep': 2}}
hmm_stat['label_actual_color'] = hmm_stat['label']
hmm_stat['label_actual_plot'] = hmm_stat['label']
hmm_stat = hmm_stat.replace(state2numColor)
hmm_stat = hmm_stat.replace(state2numPlot)
# generate the actual path plots
axs[0].axvline(x=2.5e6, color='black')
axs[0].plot(xs, hmm_stat['label_actual_plot'], color='lightgrey', alpha=0.5)
axs[0].scatter(xs, hmm_stat['label_actual_plot'], c=hmm_stat['label_actual_color'],
               cmap='cool', edgecolor='none', s=size, alpha=0.7)

''' Viterbi Path '''
# relabeling strings as numbers for plotting purposes
state2numColor = {'viterbi_class_xpehh_color': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
state2numPlot = {'viterbi_class_xpehh_plot': {'neutral': 0, 'link_left': 1, 'link_right': 1, 'sweep': 2}}
hmm_stat['viterbi_class_xpehh_plot'] = hmm_stat['viterbi_class_xpehh']
hmm_stat['viterbi_class_xpehh_color'] = hmm_stat['viterbi_class_xpehh']
hmm_stat = hmm_stat.replace(state2numColor)
hmm_stat = hmm_stat.replace(state2numPlot)
# generate the viterbi path plots
axs[1].axvline(x=2.5e6, color='black')
axs[1].plot(xs, hmm_stat['viterbi_class_xpehh_plot'], color='lightgrey', alpha=0.5)
axs[1].scatter(xs, hmm_stat['viterbi_class_xpehh_plot'], c=hmm_stat['viterbi_class_xpehh_color'],
               cmap='cool', edgecolor='none', s=size, alpha=0.7)

''' Stochastic Backtrace Paths'''
# relabeling strings as numbers for plotting purposes (bringing outside the loop to accelerate run time)
col_list = hmm_stat.columns
sb_list = [i for i in col_list if 'sb_' in i]
sb_color = hmm_stat[sb_list].replace(['neutral', 'link_left', 'link_right', 'sweep'],
                                 [0, 1, 2, 3])
sb_plot = hmm_stat[sb_list].replace(['neutral', 'link_left', 'link_right', 'sweep'],
                                 [0, 1, 1, 3])
# generate the stochastic backtrace plots
for i in sb_list:
    axs[2].plot(xs, sb_plot[i], color='lightgrey', alpha=0.1)
    axs[2].scatter(xs, sb_plot[i], c=sb_color[i], cmap='cool', edgecolor='none', s=size, alpha=0.7)
axs[2].axvline(x=2.5e6, color='black')

''' SWIFr Plot One Stat '''
# Not all classes are always identified with swifr, therefore, adjust the colorbar accordingly
maxval = (len(swifr_select['label_pred'].unique()))/(len(swifr_select_classes))
# truncate_colormap function found in stack exchange (link in function)
new_cmap = truncate_colormap(plt.get_cmap('cool'), minval=0, maxval=maxval, n=100)
# relabeling strings as numbers for plotting purposes
state2numColor = {'label_pred_color': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
state2numPlot = {'label_pred_plot': {'neutral': 0, 'link_left': 1, 'link_right': 1, 'sweep': 2}}
swifr_select['label_pred_color'] = swifr_select['label_pred']
swifr_select['label_pred_plot'] = swifr_select['label_pred']
swifr_select = swifr_select.replace(state2numColor)
swifr_select = swifr_select.replace(state2numPlot)
# generate SWIFr plots
axs[3].axvline(x=2.5e6, color='black')
axs[3].plot(xs2, swifr_select['label_pred_plot'], color='lightgrey', alpha=0.5)
axs[3].scatter(xs2, swifr_select['label_pred_plot'], c=swifr_select['label_pred_color'], cmap=new_cmap, edgecolor='none', s=size)

''' SWIFr Plot All Stats '''
# Not all classes are always identified with swifr, therefore, adjust the colorbar accordingly
maxval = (len(sall['label_pred'].unique()))/(len(sxpehh_classes))
# truncate_colormap function found in stack exchange (link in function)
new_cmap = truncate_colormap(plt.get_cmap('cool'), minval=0, maxval=maxval, n=100)
# relabeling strings as numbers for plotting purposes
state2numColor = {'label_pred_color': {'neutral': 0, 'link_left': 1, 'link_right': 2, 'sweep': 3}}
state2numPlot = {'label_pred_plot': {'neutral': 0, 'link_left': 1, 'link_right': 1, 'sweep': 2}}
sall['label_pred_color'] = sall['label_pred']
sall['label_pred_plot'] = sall['label_pred']
sall = sall.replace(state2numColor)
sall = sall.replace(state2numPlot)
# generate SWIFr plots
axs[4].axvline(x=2.5e6, color='black')
axs[4].plot(xs3, sall['label_pred_plot'], color='lightgrey', alpha=0.5)
axs[4].scatter(xs3, sall['label_pred_plot'], c=sall['label_pred_color'], cmap=new_cmap, edgecolor='none', s=size)

''' Plot of Input Statistic'''
axs[5].axvline(x=2.5e6, color='black')
stats_plot = axs[5].scatter(xs, hmm_stat[statistic], c=hmm_stat[statistic], cmap='cool', s=3)

# Add colorbars inside the stats plot (bottom plot)
cax1 = inset_axes(axs[5], width="2%", height="100%", loc='right', borderpad=0)
cbar5 = fig.colorbar(stats_plot, cax=cax1)

''' Plot settings '''
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

axs[3].set(ylabel='SWIFr 1-Stat')
axs[3].yaxis.set_ticks(axis_ticks)
axs[3].set_yticklabels(axis_labels)

axs[4].set(ylabel='SWIFr All-Stats')
axs[4].yaxis.set_ticks(axis_ticks)
axs[4].set_yticklabels(axis_labels)

axs[5].set(ylabel='Value XP-EHH')
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
# plt.savefig('plots_thesis/path_plot_4class_xpehh.svg')
plt.show()






