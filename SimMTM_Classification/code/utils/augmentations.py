import numpy as np
import torch
import math


def data_transform_masked4cl(sample, masking_ratio, lm, positive_nums=None, distribution='geometric'):
    """Masked time series in time dimension"""

    if positive_nums is None:
        positive_nums = math.ceil(1.5 / (1 - masking_ratio))

    sample = sample.permute(0, 2, 1)

    sample_repeat = sample.repeat(positive_nums, 1, 1)

    mask = noise_mask(sample_repeat, masking_ratio, lm, distribution=distribution)
    x_masked = mask * sample_repeat

    return x_masked.permute(0, 2, 1), mask.permute(0, 2, 1)


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def noise_mask(X, masking_ratio=0.25, lm=3, distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        mask = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2], lm, masking_ratio)
        mask = mask.reshape(X.shape[0], X.shape[1], X.shape[2])
    elif distribution == 'masked_tail':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * (1 - masking_ratio))
            keep_mask[:, :n] = True
            mask[m, :] = keep_mask  # time dimension
    elif distribution == 'masked_head':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * masking_ratio)
            keep_mask[:, n:] = True
            mask[m, :] = keep_mask  # time dimension
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                p=(1 - masking_ratio, masking_ratio))
    return torch.tensor(mask)