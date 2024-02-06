"""
This simulator is super basic and intended to generate a string of neutral and positive events. For now this
will be set up as 0's and 1', where a vast majority of the events are 0's:
What is needed?
- Coin_1 --> Fires off 99% of the time in the range N(mu=3, sd=2)...or whatever
- Coin_2 --> Fires off 1% of the time in the range N(mu=5, sd=1)...or whatever

Using the simple_sim output, I need to:
- Use a HMM to determine if coin_1 (neutral) or coin_2 (sweep) was used
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

save_data = 'no'
samples = 2000
mu_n = 3  # mean for neutral state normal distribution
sd_n = 2  # sd for neutral state normal distribution
mu_p = 8  # mean for positive state normal distribution
sd_p = 2  # sd for positive state normal distribution

data = pd.DataFrame(columns=['state', 'label', 'value'])
a_tran = np.array([[0.95, 0.05], [0.2, 0.8]])

'''Initiate with a neutral state'''
state = []; label = []; coin = []
state.append(1)
label.append('N')
coin.append(np.random.normal(loc=mu_n, scale=sd_n, size=1)[0])

for i in range(1, samples):
    if state[i-1] == 1:  # if the state is neutral
        draw = np.random.uniform(low=0, high=1, size=1)[0]
        if draw <= a_tran[0, 0]:  # if yes, then we stay in the neutral state
            state.append(1)
            label.append('N')
            coin.append(np.random.normal(loc=mu_n, scale=sd_n, size=1)[0])
        else:  # else we bump to the positive state
            state.append(0)
            label.append('P')
            coin.append(np.random.normal(loc=mu_p, scale=sd_p, size=1)[0])

    elif state[i - 1] == 0:  # if the state is not positive, then it must be negative (condition included anyway)
        draw = np.random.uniform(low=0, high=1, size=1)[0]
        if draw <= a_tran[1, 1]:  # if yes, then we stay in the positive state
            state.append(0)
            label.append('P')
            coin.append(np.random.normal(loc=mu_p, scale=sd_p, size=1)[0])
        else:  # else we bump to the negative state
            state.append(1)
            label.append('N')
            coin.append(np.random.normal(loc=mu_n, scale=sd_n, size=1)[0])

data['state'] = state
data['label'] = label
data['value'] = coin
data = data.reset_index().rename(columns={'index': 'snp'})

neutral = data[data['state'] == 1]
positive = data[data['state'] == 0]

print("Count of Neutral SNPs: ", len(neutral))
print("Count of Positive SNPs: ", len(positive))
print("Percent of Neutral SNPs: ", round(len(neutral) / len(data) * 100, 2), ' %')

if save_data == 'yes':
    data.to_csv('output/hmm.csv', index=False)
"""
Next steps:
1. For now we will assume the distributions are known as the first objective is to recreate the N,P string using HMM
2. Therefore, import the output from this module into a separate file that performs HMM using python
3. Then write HMM code in a fashion similar to the final (but used packages where possible)
"""

# check the data
plt.figure()
plt.hist(data['value'], bins=50, color='black', alpha=0.3)
plt.hist(neutral['value'], bins=50, color='red', alpha=0.4)
plt.hist(positive['value'], bins=50, color='blue', alpha=0.4)
plt.show()

if samples < 4000:
    plt.figure()
    plt.bar(x=neutral['snp'], height=neutral['state']-1, color='red')
    plt.bar(x=positive['snp'], height=positive['state']+1, color='blue')
    plt.show()
