import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def build_spatial_adj(adata):
    try:
        import squidpy as sq
    except ImportError as exc:
        raise ImportError('Install stOTopic[spatial] for SVG selection or SGC') from exc
    coords = np.asarray(adata.obsm['spatial'])
    if coords.ndim != 2 or coords.shape[1] < 2 or (not np.isfinite(coords).all()):
        raise ValueError('Expected finite spatial coordinates')
    is_grid = np.allclose(coords, np.round(coords)) and np.ptp(coords) < 1000
    if is_grid:
        sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=6)
    else:
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)
    return adata.obsp['spatial_connectivities'].astype('float32')

def prepare(adata, *, n_top_genes=6000, n_components=400, gene_selection='hvg', min_counts=50, min_cells=None, use_sgc=True, sgc_layers=1, sgc_mode='sign', sgc_add_diag=True, random_state=None):
    if gene_selection not in {'hvg', 'svg', 'all'}:
        raise ValueError("gene_selection must be 'hvg', 'svg', or 'all'")
    if n_top_genes < 1 or n_components < 1 or min_counts < 0:
        raise ValueError('Gene/PCA counts must be positive and min_counts nonnegative')
    if 'log1p' in adata.uns:
        raise ValueError('prepare expects raw counts; provide a count AnnData before log1p')
    data = adata.copy()
    if not data.var_names.is_unique or not data.obs_names.is_unique:
        raise ValueError('Observation and gene names must be unique')
    values = data.X.data if issparse(data.X) else np.asarray(data.X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('X must contain finite nonnegative counts')
    if not np.allclose(values, np.round(values)):
        raise ValueError('prepare expects raw integer counts, not normalized expression')
    sc.pp.filter_cells(data, min_counts=min_counts)
    if min_cells is None:
        min_cells = max(1, int(data.n_obs * 0.01))
    if min_cells < 1:
        raise ValueError('min_cells must be positive')
    sc.pp.filter_genes(data, min_cells=min_cells)
    if data.n_obs < 2 or data.n_vars < 2:
        raise ValueError('Filtering left fewer than two cells or genes')
    if gene_selection == 'hvg' and n_top_genes < data.n_vars:
        sc.pp.highly_variable_genes(data, n_top_genes=n_top_genes, flavor='seurat_v3')
        data = data[:, data.var['highly_variable']].copy()
    elif gene_selection == 'svg':
        import squidpy as sq
        normalized = data.copy()
        sc.pp.normalize_total(normalized, target_sum=10000.0)
        sc.pp.log1p(normalized)
        build_spatial_adj(normalized)
        sq.gr.spatial_autocorr(normalized, mode='moran', n_perms=1000, n_jobs=1, seed=random_state)
        results = normalized.uns['moranI']
        genes = results.loc[results['pval_norm_fdr_bh'] < 0.05].sort_values('I', ascending=False).head(n_top_genes).index
        if len(genes) < 2:
            raise ValueError('Fewer than two significant spatially variable genes')
        data = data[:, genes].copy()
    normalized = data.copy()
    sc.pp.normalize_total(normalized, target_sum=10000.0)
    sc.pp.log1p(normalized)
    features = normalized.X.toarray() if issparse(normalized.X) else np.asarray(normalized.X)
    features = features.astype(np.float32)
    if use_sgc:
        from ._sgc import precompute_SGC_scipy
        import torch
        if sgc_mode not in {'sgc', 'sign'} or sgc_layers < 0:
            raise ValueError('SGC mode must be sgc/sign and layers must be nonnegative')
        adj = build_spatial_adj(data)
        features = precompute_SGC_scipy(torch.from_numpy(features), adj, sgc_layers, mode=sgc_mode, add_diag=sgc_add_diag).numpy()
    dimensions = min(n_components, data.n_obs - 1, features.shape[1] - 1)
    reducer = PCA(n_components=dimensions, random_state=random_state)
    embeddings = normalize(reducer.fit_transform(features), norm='l2', axis=1).astype(np.float32)
    data.obsm['X_pca'] = embeddings
    data.uns['stotopic_preprocessing'] = {'gene_selection': gene_selection, 'n_components': dimensions, 'use_sgc': use_sgc, 'sgc_layers': sgc_layers, 'sgc_mode': sgc_mode, 'explained_variance_ratio': float(reducer.explained_variance_ratio_.sum())}
    return (data, embeddings)
