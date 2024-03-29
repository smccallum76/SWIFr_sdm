import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
---------------------------------------------------------------------------------
Import and Sorting
---------------------------------------------------------------------------------
"""
stat = 'ihs_afr_std'
# import
data = pd.read_csv(f'output/stochastic_bt_1000_sims_{stat}.csv')
# sorting
data = data.sort_values(by='snp_position').reset_index(drop=True)

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
ihs_slim = data[data_cols2]

"""
---------------------------------------------------------------------------------
Define bins
---------------------------------------------------------------------------------
"""
bp_per_bin = 50000
max_bin = np.max(ihs_slim['snp_position'])
bin_count = int(np.ceil(max_bin/bp_per_bin))
sweeps = ihs_slim.groupby(pd.cut(ihs_slim['snp_position'], bins=bin_count), observed=False).sum()['n_sweepLike']

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


plt.figure(figsize=(15, 8))
plt.bar(x=idx2[min_x:max_x], height=sweeps[min_x:max_x]/np.sum(sweeps), color=colors)
plt.xticks(rotation='vertical')
plt.xlabel("Genomic Position")
plt.ylabel("Sweep and Link Frequency")
plt.title(f"Frequency of Sweep and Link Events, Stat = {stat}")
plt.show()

print('done')