from itertools import combinations
import numpy as np
import pandas as pd
from scipy.sparse import issparse

def _top_genes(beta, topk):
    if not isinstance(beta, pd.DataFrame) or beta.empty:
        raise ValueError('beta must be a nonempty genes x topics DataFrame')
    if not beta.index.is_unique or not beta.columns.is_unique:
        raise ValueError('Gene and topic names must be unique')
    if not np.isfinite(beta.to_numpy()).all() or (beta.to_numpy() < 0).any():
        raise ValueError('beta must contain finite, nonnegative weights')
    if not isinstance(topk, int) or topk < 1:
        raise ValueError('topk must be a positive integer')
    return [beta.nlargest(min(topk, len(beta)), col).index.tolist() for col in beta]

def get_topic_diversity(beta, topk=20, p=0.9):
    topics = _top_genes(beta, topk)
    if not 0 < p < 1:
        raise ValueError('p must lie strictly between 0 and 1')
    result = np.ones(len(topics))
    for (i, j) in combinations(range(len(topics)), 2):
        overlap = sum((len(set(topics[i][:d]) & set(topics[j][:d])) / d * p ** (d - 1) for d in range(1, len(topics[i]) + 1)))
        dissimilarity = 1 - (1 - p) * overlap
        result[i] = min(result[i], dissimilarity)
        result[j] = min(result[j], dissimilarity)
    return result.tolist()

def get_topic_redundancy(beta, topk=20):
    topics = _top_genes(beta, topk)
    if len(topics) < 2:
        raise ValueError('At least two topics are required')
    if topk > len(beta):
        raise ValueError(f'topk={topk} exceeds n_genes={len(beta)}')
    sets = [set(genes) for genes in topics]
    return float(np.mean([len(a & b) for a, b in combinations(sets, 2)]))

def get_topic_coherence(adata, beta, layer=None, topk=20, quantile=0.75, individual=False):
    topics = _top_genes(beta, topk)
    if not 0 <= quantile <= 1 or adata.n_obs == 0:
        raise ValueError('quantile must be in [0, 1] and adata must contain cells')
    if not adata.var_names.is_unique:
        raise ValueError('adata.var_names must be unique')
    genes = list(dict.fromkeys((g for topic in topics for g in topic)))
    indices = adata.var_names.get_indexer(genes)
    if (indices < 0).any():
        raise ValueError('Some beta genes are missing from adata.var_names')
    matrix = adata.X if layer is None else adata.layers[layer]
    matrix = matrix[:, indices]
    matrix = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('Expression values must be finite and nonnegative')
    detected = matrix >= np.maximum(np.quantile(matrix, quantile, axis=0), 1)
    lookup = {g: i for (i, g) in enumerate(genes)}
    probabilities = detected.mean(axis=0)
    scores = []
    for topic in topics:
        pairs = []
        for (a, b) in combinations(topic, 2):
            (i, j) = (lookup[a], lookup[b])
            joint = np.mean(detected[:, i] & detected[:, j])
            score = 0.0 if joint == 0 or joint == 1 else (np.log2(joint) - np.log2(probabilities[i]) - np.log2(probabilities[j])) / -np.log2(joint)
            pairs.append(score)
        scores.append(float(np.mean(pairs)) if pairs else 0.0)
    return scores if individual else float(np.mean(scores))
