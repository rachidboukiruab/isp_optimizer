import torch


def norm_hyperparameter(bounds, hyp):
    lowest_value = bounds[0]
    highest_value = bounds[1]
    b = highest_value - lowest_value
    norm_hyp = (hyp - lowest_value)/b
    return norm_hyp


def denorm_hyperparameter(bounds, norm_hyp):
    lowest_value = bounds[0]
    highest_value = bounds[1]
    b = highest_value - lowest_value
    hyp = norm_hyp*b + lowest_value
    return hyp


def collate_fn(batch):
    return tuple(zip(*batch))