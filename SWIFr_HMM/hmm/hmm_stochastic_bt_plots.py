import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import hmm_funcs as hmm

"""
---------------------------------------------------------------------------------
Import and Sorting
---------------------------------------------------------------------------------
"""
stat = 'fst'
classes = 4
# import
data = pd.read_csv(f'output/stochastic_bt_1000_sims_{stat}.csv')
# sorting
data = data.sort_values(by='snp_position').reset_index(drop=True)

if stat == 'ihs_afr_std':
    swifr_path_1stat = '../../swifr_pkg/test_data/simulations_4_swifr_test_4class_ihs/test/test_classified'
elif stat == 'xpehh':
    swifr_path_1stat = '../../swifr_pkg/test_data/simulations_4_swifr_test_4class_xpehh/test/test_classified'
elif stat == 'fst':
    swifr_path_1stat = '../../swifr_pkg/test_data/simulations_4_swifr_test_4class_fst/test/test_classified'
swfr_classified = hmm.hmm_get_data(swifr_path_1stat)
'''
SWIFr scenarios below were trained and run on only one stat at a time. This was done for direct comparison
with the HMM trials using only one stat at a time.
'''
swfr_classified_1stat = hmm.hmm_get_data(swifr_path_1stat)
swfr_classified_1stat = swfr_classified_1stat[swfr_classified_1stat[f'{stat}'] != -998].reset_index(drop=True)

# convert swifr probabilities into a classification code
swifr_cols = [
                'P(neutral)',
                'P(link_left)',
                'P(link_right)',
                'P(sweep)'
            ]
# find the largest prob
swfr_classified_1stat['swfr_class'] = swfr_classified_1stat[swifr_cols].idxmax(axis='columns')
# create a new class column and then replace the strings with numbers
swfr_classified_1stat['swfr_class_num'] = swfr_classified_1stat['swfr_class']
swfr_classified_1stat['swfr_class_num'] = swfr_classified_1stat['swfr_class_num'].replace(swifr_cols, [0, 1, 2, 3])

"""
---------------------------------------------------------------------------------
Count occurrences at each SNP site
---------------------------------------------------------------------------------
"""
data_cols = [i for i in data.columns if 'sb_' in i]
data['n_neutral'] = np.nansum(data[data_cols] == 0, axis=1)
data['n_link_left'] = np.nansum(data[data_cols] == 1, axis=1)
data['n_link_right'] = np.nansum(data[data_cols] == 2, axis=1)
data['n_sweep'] = np.nansum(data[data_cols] == 3, axis=1)
data['n_sweepLike'] = np.sum(data[['n_link_right', 'n_link_left', 'n_sweep']], axis=1)

data_cols2 = [i for i in data.columns if 'sb_' not in i]
data_slim = data[data_cols2]

"""
---------------------------------------------------------------------------------
Define bins
---------------------------------------------------------------------------------
"""
bp_per_bin = 50000
max_bin = np.max(data_slim['snp_position'])
bin_count = int(np.ceil(max_bin/bp_per_bin))
sweeps = data_slim.groupby(pd.cut(data_slim['snp_position'], bins=bin_count), observed=False).sum()['n_sweepLike']
xx = np.sum(sweeps)

idx = sweeps.index.values.categories.mid.astype(int).tolist()
idx2 = [str(round(i, -4)) for i in idx]
sweeps = np.asarray(sweeps, int)
"""
---------------------------------------------------------------------------------
Plots
---------------------------------------------------------------------------------
"""

min_x = int(len(idx) * .2)
max_x = int(len(idx) * .8)
colors = ['tomato' if x > 2450000 and x < 2550000 else 'dodgerblue'for x in idx[min_x:max_x]]

custom_lines = [Line2D([0], [0], color='dodgerblue', lw=4),
                Line2D([0], [0], color='tomato', lw=4)]

plt.figure(figsize=(15, 8))
plt.bar(x=idx2[min_x:max_x], height=sweeps[min_x:max_x]/np.sum(sweeps), color=colors)
plt.xticks(rotation='vertical')
plt.xlabel("Genomic Position")
plt.ylabel("Sweep and Link Frequency [non-neutral events]")
plt.title(f"Frequency of Sweep and Link Events, Stat = {stat}")
plt.legend(custom_lines, ['Non-Neutral', 'Sweep Region'])
plt.show()

"""
---------------------------------------------------------------------------------------------------
Plot -- Path and stat comparison [flashlight plot]
---------------------------------------------------------------------------------------------------
"""
data2 = data.sort_values(by='idx_key')
xs = np.arange(0, len(data2), 1)
xs2 = np.arange(0, len(swfr_classified_1stat), 1)

cmap = mpl.colormaps['viridis']
legend_colors = cmap(np.linspace(0, 1, classes))

fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(5, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle(f'Actual Path, Predicted Path, and {stat}')
axs[0].plot(xs, data2['label_num'], color='black')
axs[0].scatter(xs, data2['label_num'], c=data2['label_num'],
               cmap='viridis', edgecolor='none', s=30)

axs[1].plot(xs, data2[f'viterbi_class_{stat}'], color='black')
axs[1].scatter(xs, data2[f'viterbi_class_{stat}'], c=data2[f'viterbi_class_{stat}'],
               cmap='viridis', edgecolor='none', s=30)
for i in range(len(data_cols)-990):
    axs[2].plot(xs, data2[data_cols[i]], color='lightgrey')

axs[3].plot(xs2, swfr_classified_1stat['swfr_class_num'], color='black')
axs[3].scatter(xs2, swfr_classified_1stat['swfr_class_num'], c=swfr_classified_1stat['swfr_class_num'],
               cmap='viridis', edgecolor='none', s=30)

axs[4].scatter(xs, data2[f'{stat}'], c=data2[f'{stat}'], cmap='viridis', edgecolor='none', s=3)

axs[0].set(ylabel='Actual State')
axs[1].set(ylabel='Viterbi Pred')
axs[2].set(ylabel='BackTrc Pred')
axs[3].set(ylabel='SWIFr Pred')
axs[4].set(ylabel=f'Value {stat}')
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

plt.show()

print('done')