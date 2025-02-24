import numpy as np

def normalize_state(state):
    """Normalize the quantum state."""
    norm = np.linalg.norm(state)
    if norm == 0:
        return state
    return state / norm
