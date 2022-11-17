import numpy as np


def transform_target(target, num_classes, epsilon=0.01):
    w = np.ones(shape=(num_classes))
    label_one = np.zeros(shape=(num_classes))

    label_one[target] += 1
    labels_out = (1 - epsilon) * label_one + epsilon * w / num_classes
    return labels_out

def transform_trace(trace, h_size=32):
    x_new = np.zeros(shape=(h_size, trace.shape[1]))
    x_new[h_size // 2] = trace
    return x_new[np.newaxis, :]