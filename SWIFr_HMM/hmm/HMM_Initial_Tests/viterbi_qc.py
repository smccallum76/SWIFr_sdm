import numpy as np

def hmm_viterbi(a, pi_viterbi, bx_obs):
    # some initializations and settings
    n = len(bx_obs[0, :])
    classes = ['A', 'B']
    delta = np.empty(shape=(len(classes), n))
    bx = bx_obs  # equivalent to the observations from hannah's script
    pointer = np.empty(shape=(len(classes), n)) + -np.inf
    path = np.empty(shape=(n,))

    # take the log of all the Viterbi required properties
    for c in range(len(classes)):
        bx[c] = np.log(bx[c])
        pi_viterbi[c] = np.log(pi_viterbi[c])

    # a = np.log(a)
    # loop to catch div by zero warnings
    for ci in range(len(classes)):
        for cj in range(len(classes)):
            if (a[ci, cj] == 0) or (a[ci, cj] == -np.inf):
                a[ci, cj] = -np.inf
            else:
                a[ci, cj] = np.log(a[ci, cj])

    # initiate the sequence for each class at the 0th index
    for c in range(len(classes)):
        delta[c, 0] = pi_viterbi[c] + bx[c][0]

    # pointer[:, 0] = np.argmax(p[:, 0], axis=0)
    pointer[:, 0] = np.nan
    for t in range(1, n):
        # pointer points to the previous most state that was most likely (where we are actually moving from).
        # this is needed b/c when the transition matrix has zero prob, there are impossible moves.

        for ci in range(len(classes)):
            temp = []
            for cj in range(len(classes)):
                # temp hold the values intermediate values that evaluate the probability of being in the current
                # class when considering all possible transitions to this class.
                temp.append(delta[cj, t - 1] + a[cj, ci])
            delta[ci, t] = bx[ci][t] + np.max(temp)
            pointer[ci, t] = np.argmax(temp) # this points to the previous state the leads to the current state with highest prob

    # backtrace using the pointer recorded above
    path[n-1] = np.argmax(delta[:, n-1], axis=0)
    for t in reversed(range(n-1)):
        xx = int(path[t+1])
        path[t] = pointer[int(path[t+1]), t+1]

    return path, delta
"""
2, 4
A, B
A, T, G, C  # these are the observation values
0.5, 0.5  # pi values for hidden state A and B
0.75, 0.25; 0.25, 0.75  # Transition matrix values
0.3, 0.3, 0.2, 0.2; 0.25, 0.25, 0.25, 0.25  # props of each observation in each hidden state

Observations input
ATATAAAGCACCGTTGCG
Correct Viterbi path output
AAAAAAABBBBBBBBBBB
Correct Viterbi probability at path end (max prob at terminal point of path)
6.5325e-14
"""
obs_string = "ATATAAAGCACCGTTGCG" # user input
obs = [c for c in obs_string]  # convert user input to a list
pi = np.array([0.5, 0.5])  # prior probabilities
a_trans = np.array([[0.75, 0.25], [0.25, 0.75]])  # transition matrix
# probs in order A, T, G, C. For example, the first value of 'A':0.3 indicates that the prob of seeing an A-protein in
# hidden stat 'A' (b/c first row is state A) is 30% (confusing with states and string the same name). The value directly
# beneath is 'A':0.25, which indicates that the probability of seeing A-protein in hidden state B is 25%.
probs = np.array([[0.3, 0.3, 0.2, 0.2], [0.25, 0.25, 0.25, 0.25]])
# list of dictionary that is used to transform the string to a probability value for each state. This is equivalent to
# the bx matrix of my code.
obs_dicts = [{'A':0.3, 'T':0.3, 'G':0.2, 'C':0.2},
             {'A':0.25, 'T':0.25, 'G':0.25, 'C':0.25}]

# create the bx matrix based on the obs_dict
bx_obs = np.empty(shape=(len(pi), len(obs)))
for i in range(len(pi)):
    for j in range(len(obs)):
        bx_obs[i, j] = obs_dicts[i][obs[j]]

v_path, v_delta = hmm_viterbi(a_trans, pi, bx_obs)
v_delta = np.exp(v_delta)
print('Path --> ', v_path)
print('Max Prob at path end: ', np.max(v_delta[:, len(obs)-1]))