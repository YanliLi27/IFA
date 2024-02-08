import numpy as np


def stat_calculator(cam_grad_max_matrix, cam_grad_min_matrix):
    grad_max_percentile = np.percentile(cam_grad_max_matrix, (25, 50, 75, 90, 99))
    grad_min_percentile = np.percentile(cam_grad_min_matrix, (25, 50, 75, 90, 99))
    return grad_max_percentile, grad_min_percentile