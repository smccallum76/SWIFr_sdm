"""
Using the example PDF in my thesis drive from U. Penn (Example-Viterbi-DNA)
"""
import numpy as np
# probability of being in state H or state L
pi = np.log2(np.array([0.5, 0.5]))
# transition matrix for moving between stats H and L
A = np.log2(np.array([[0.5, 0.5], [0.4, 0.6]]))
# probability of taking A, C, G, T given the state is H
H = np.log2(np.array([0.2, 0.3, 0.3, 0.2]))
# probability of taking A, C, G, T given the state is L
L = np.log2(np.array([0.3, 0.2, 0.2, 0.3]))

# S is the observed sequence of events
# A=0, C=1, G=2, T=3
S = np.array([2, 2, 1, 0, 1, 3, 2, 0, 0])  # equates to [G, G, C, A, C, T, G, A, A]
p_h = []  # list to hold the log probs of being in state H
p_l = []  # list to hold the log probs of being in state L

'''
We loop over the length of the Sequence and determine the log probs for each state to yield each element of the 
sequence
'''
for i in range(len(S)):
    if i == 0:
        # The first element (0th position)
        p_h.append(pi[0] + H[S[i]])
        p_l.append(pi[1] + L[S[i]])
    else:
        # The second element
        p_h.append(H[S[i]] + np.maximum(p_h[i-1] + A[0, 0], p_l[i-1] + A[1, 0]))
        p_l.append(L[S[i]] + np.maximum(p_l[i-1] + A[1, 1], p_h[i-1] + A[0, 1]))

# combine all the states into a single array, where each state is one row
p_h_l = np.array([p_h, p_l])
# return the index of the max value in each column which corresponds to states 0, 1, ...
# note that in the event of a tie the first index is returned. However, in a real situation a tie would be extremely
# unlikely.
path = np.argmax(p_h_l, axis=0)

print(p_h)
print(p_l)
print(path)




