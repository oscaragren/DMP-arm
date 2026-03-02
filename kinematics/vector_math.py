import numpy as np

def build_stack(v1, v2):
    return np.stack([v1, v2], axis=1)

a1 = np.array([1, 0, 0])
a2 = np.array([0, 1, 0])
a3 = np.array([0, 0, 1])

v1, v2 = build_stack(a1, a2)

print(v1)