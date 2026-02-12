import numpy as np


def get_cosine_similarity(v1, v2):
    # dot product of the two vectors
    dot_product = np.dot(v1, v2)

    # magnitudes (norms) of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    return dot_product / (norm_v1 * norm_v2)


# Test it
vec_a = np.array([1, 2, 3])
vec_b = np.array([1, 2, 2])  # very similar to vec_a
print(f"Similarity: {get_cosine_similarity(vec_a, vec_b)}")
