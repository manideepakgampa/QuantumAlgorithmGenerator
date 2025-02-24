import numpy as np

def measure(state_vector):
    probabilities = np.abs(state_vector) ** 2
    outcome = np.random.choice(len(state_vector), p=probabilities)
    return format(outcome, 'b').zfill(int(np.log2(len(state_vector))))
