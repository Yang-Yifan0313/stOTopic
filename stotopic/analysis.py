import numpy as np
import pandas as pd

LUNG_PROGRAMS = {
    'Hypoxia-associated': ['NDRG1', 'VEGFA', 'SLC2A1', 'HILPDA', 'IGFBP5'],
    'Proliferation-associated': ['MKI67', 'TOP2A', 'UBE2C', 'CENPF', 'PTTG1', 'TYMS', 'BIRC5'],
}


def validate_beta(beta):
    if not isinstance(beta, pd.DataFrame) or beta.empty:
        raise ValueError('beta must be a nonempty genes-by-topics DataFrame')
    if not beta.index.is_unique or not beta.columns.is_unique:
        raise ValueError('Gene and topic names must be unique')
    if not np.isfinite(beta.to_numpy()).all() or (beta.to_numpy() < 0).any():
        raise ValueError('Gene weights must be finite and nonnegative')
    return beta


def top_genes(beta, top_n=10):
    validate_beta(beta)
    if top_n < 1:
        raise ValueError('top_n must be positive')
    rows = []
    for topic in beta.columns:
        for rank, (gene, weight) in enumerate(beta[topic].nlargest(top_n).items(), 1):
            rows.append({'topic': topic, 'rank': rank, 'gene': gene, 'weight': float(weight)})
    return pd.DataFrame(rows)


def summarize_topics(topic_prop, beta, top_n=10):
    validate_beta(beta)
    if not isinstance(topic_prop, pd.DataFrame) or not topic_prop.columns.equals(beta.columns):
        raise ValueError('Topic columns must match between theta and beta')
    values = topic_prop.to_numpy()
    if values.shape[0] == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('theta must contain finite nonnegative proportions')
    if not np.allclose(values.sum(axis=1), 1, atol=1e-5):
        raise ValueError('theta rows must sum to one')
    dominant = np.argmax(values, axis=1)
    return pd.DataFrame({
        'topic': beta.columns,
        'mean_proportion': values.mean(axis=0),
        'dominant_cells': np.bincount(dominant, minlength=values.shape[1]),
        'top_genes': [', '.join(beta[t].nlargest(top_n).index.astype(str)) for t in beta.columns],
    })


def celltype_activity(adata, topic_prop, cell_type_key):
    from .pl import _topic_data
    _topic_data(adata, topic_prop)
    if cell_type_key not in adata.obs:
        raise ValueError('Cell-type field not found: ' + cell_type_key)
    labels = adata.obs[cell_type_key]
    valid = labels.notna()
    if not valid.any():
        raise ValueError('Cell-type annotations contain no nonmissing labels')
    groups = labels.loc[valid].astype(str)
    frame = topic_prop.loc[valid]
    mean = frame.groupby(groups, sort=True).mean()
    onehot = pd.DataFrame(np.eye(frame.shape[1])[frame.to_numpy().argmax(axis=1)], index=frame.index, columns=frame.columns)
    fraction = onehot.groupby(groups, sort=True).mean()
    return mean, fraction


def select_programs(beta, signatures=None, top_n=20, min_markers=2):
    validate_beta(beta)
    signatures = LUNG_PROGRAMS if signatures is None else signatures
    ranks = beta.rank(axis=0, ascending=False, method='first')
    selections = []
    for name, genes in signatures.items():
        available = [g for g in genes if g in beta.index]
        candidates = []
        for topic in beta.columns:
            matched = sorted([g for g in available if ranks.loc[g, topic] <= top_n], key=lambda g: ranks.loc[g, topic])
            if len(matched) >= min_markers:
                candidates.append((len(matched), sum(1.0 / ranks.loc[g, topic] for g in matched), topic, matched))
        candidates.sort(key=lambda row: (-row[0], -row[1], str(row[2])))
        selections.append((name, available, candidates))
    result = []
    for name, available, candidates in selections:
        if not candidates:
            result.append({'program': name, 'topic': None, 'available_markers': available, 'matched_markers': [], 'status': 'insufficient_top_gene_support'})
        else:
            chosen = candidates[0]
            result.append({'program': name, 'topic': chosen[2], 'available_markers': available, 'matched_markers': chosen[3], 'status': 'marker_supported_candidate'})
    chosen_topics = [row['topic'] for row in result if row['topic'] is not None]
    if len(set(chosen_topics)) < len(chosen_topics):
        for row in result:
            if chosen_topics.count(row['topic']) > 1:
                row['status'] = 'shared_topic_no_separate_programs'
    return result
