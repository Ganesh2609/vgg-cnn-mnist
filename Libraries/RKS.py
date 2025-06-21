# Decreases accuracy, hence not used
import numpy as np

def map_rks(data, n):
    m = len(data[list(data.keys())[0]][0])
    random_matrix = np.random.randint(-9, 10, size=(n, m))
    out = {}
    
    for key in data:
        arr = data[key]
        out[key] = []
        for vector in arr:
            vector = np.array(vector)
            temp = (random_matrix @ vector)
            cos_values = list(np.cos(temp))
            sin_values = list(np.sin(temp))
            new = cos_values
            new.extend(sin_values)
            out[key].append(new)

    return out            