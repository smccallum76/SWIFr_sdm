import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
---------------------------------------------------------------------------------
Import and Sorting
---------------------------------------------------------------------------------
"""
# import
ihs = pd.read_csv('output/stochastic_bt_1000_sims_ihs_afr_std.csv')
# sorting
ihs = ihs.sort_values(by='snp_position').reset_index(drop=True)

"""
---------------------------------------------------------------------------------
Count occurrences at each SNP site
---------------------------------------------------------------------------------
"""
ihs_cols = [i for i in ihs.columns if 'sb_' in i]
ihs['n_neutral'] = np.nansum(ihs[ihs_cols] == 0, axis=1)
ihs['n_link_left'] = np.nansum(ihs[ihs_cols] == 1, axis=1)
ihs['n_link_right'] = np.nansum(ihs[ihs_cols] == 2, axis=1)
ihs['n_sweep'] = np.nansum(ihs[ihs_cols] == 3, axis=1)
ihs['n_sweepLike'] = np.sum(ihs[['n_link_right', 'n_link_left', 'n_sweep']], axis=1)

ihs_cols2 = [i for i in ihs.columns if 'sb_' not in i]
ihs_slim = ihs[ihs_cols2]

"""
---------------------------------------------------------------------------------
Define bins
---------------------------------------------------------------------------------
"""
bp_per_bin = 50000
max_bin = np.max(ihs_slim['snp_position'])
bin_count = int(np.ceil(max_bin/bp_per_bin))
sweeps = ihs_slim.groupby(pd.cut(ihs_slim['snp_position'], bins=bin_count), observed=False).sum()['n_sweepLike']

idx = sweeps.index
"""
---------------------------------------------------------------------------------
Plots
---------------------------------------------------------------------------------
"""
plt.figure()
plt.bar(x=sweeps.index.astype(str), height=sweeps)
plt.show()

print('done')