"""Corrected local adaptation of pair1/validate_code.py and pair1/gen_ind.py.

Original interfaces retained; originals are never modified or shadowed.
"""
import numpy as np


def rank_gf2(rows):
    a = np.array(rows, dtype=int, copy=True)
    if a.size == 0:
        return 0
    rank = 0
    for col in range(a.shape[1]):
        pivots = np.flatnonzero(a[rank:, col])
        if not len(pivots):
            continue
        pivot = rank + pivots[0]
        a[[rank, pivot]] = a[[pivot, rank]]
        for r in range(len(a)):
            if r != rank and a[r, col]:
                a[r] ^= a[rank]
        rank += 1
        if rank == len(a):
            break
    return rank


def gen_ind(generators):
    try:
        a = np.asarray(generators)
        if (a.ndim != 2 or not a.shape[0] or not a.shape[1]
                or a.shape[1] % 2 or not np.all((a == 0) | (a == 1))):
            return False
        a = a.astype(int)
        n = a.shape[1] // 2
        if np.any((a[:, :n] @ a[:, n:].T + a[:, n:] @ a[:, :n].T) % 2):
            return False
        return rank_gf2(a) == len(a)
    except (TypeError, ValueError):
        return False


def validate_code(generators):
    return gen_ind(generators)
