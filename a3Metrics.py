import numpy as np
import torch

def ensure_3d(x):
    """
    Force x à être [B, T, F]
    - [T]        -> [1, T, 1]
    - [T, F]     -> [1, T, F]
    - [B, T]     -> [B, T, 1]
    - [B, T, F]  -> inchangé
    """
    if torch.is_tensor(x):
        if x.dim() == 1:
            return x.unsqueeze(0).unsqueeze(-1)
        if x.dim() == 2:
            return x.unsqueeze(0)
        if x.dim() == 3:
            return x
        raise ValueError(f"Unsupported torch shape: {tuple(x.shape)}")

    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :, None]
    if x.ndim == 2:
        return x[None, :, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Unsupported numpy shape: {x.shape}")

def squeeze_last_dim(x):
    """
    Retire la dernière dim si elle vaut 1:
    - [B, T, 1] -> [B, T]
    - [T, 1]    -> [T]
    sinon inchangé.
    """
    if torch.is_tensor(x):
        return x.squeeze(-1) if (x.dim() >= 2 and x.size(-1) == 1) else x
    x = np.asarray(x)
    return np.squeeze(x, axis=-1) if (x.ndim >= 2 and x.shape[-1] == 1) else x


def calculate_proximity(x_orig, x_cf):
    """ Calcule la distance euclidienne entre l'original et le contrefactuel """
    return np.linalg.norm(x_orig - x_cf)



import numpy as np
import pandas as pd



def validity_ratio(pred_values, desired_min, desired_max):
    """
    Formule : Ratio de points dans les bornes sur tout l'horizon T.
    Fidèle à : Ratio(X') dans le papier.
    """

    input_array = np.logical_and(pred_values <= desired_max, pred_values >= desired_min)

    return input_array.mean()

def proximity_l2(x_orig, x_cf):
    """
    Formule : Moyenne des distances Euclidiennes (L2).
    Fidèle à : Proximity(X') dans le papier.
    """

    paired_distances = np.linalg.norm(x_orig - x_cf, axis=1)
    return np.mean(paired_distances)



def compactness_score(x_orig, x_cf, tol=0.01):
    """
    Proportion de points qui restent 'proches' de l'original.
    Fidèle à : Compact(X') dans le papier.
    """
    c = np.isclose(x_orig, x_cf, atol=tol)
    return c.mean()



def stepwise_validity_auc(pred_values, desired_min, desired_max):
    input_array = np.logical_and(pred_values <= desired_max, pred_values >= desired_min)
    n_samples, n_steps_total, _ = pred_values.shape
    until_steps_valid = np.zeros(n_samples)

    for i in range(n_samples):
        step_counts = 0
        for step in range(n_steps_total):
            if input_array[i, step, 0]: 
                step_counts += 1
            else:
                break 
        until_steps_valid[i] = step_counts

    if n_samples == 1:
        return until_steps_valid[0] / n_steps_total


    valid_steps = np.arange(n_steps_total + 1)
    counts = np.zeros(n_steps_total + 1)
    
    unique, count_values = np.unique(until_steps_valid, return_counts=True)
    for u, c in zip(unique, count_values):
        counts[int(u)] = c
        

    cumsum_counts = np.flip(np.cumsum(np.flip(counts)))
    

    auc = np.trapezoid(cumsum_counts / n_samples, x=valid_steps / n_steps_total)
    return auc