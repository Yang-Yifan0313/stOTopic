from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _topic_data(adata, topic_prop):
    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] must contain spatial coordinates")
    xy = np.asarray(adata.obsm['spatial'])
    if xy.ndim != 2 or xy.shape[0] != adata.n_obs or xy.shape[1] < 2:
        raise ValueError('Spatial coordinates must have shape (n_cells, >=2)')
    if isinstance(topic_prop, pd.DataFrame):
        if not topic_prop.index.equals(adata.obs_names):
            raise ValueError('topic_prop index must match adata.obs_names in the same order')
        names = list(topic_prop.columns)
    else:
        names = None
    values = np.asarray(topic_prop)
    if values.ndim != 2 or values.shape[0] != adata.n_obs or values.shape[1] == 0:
        raise ValueError('topic_prop must have shape (n_cells, n_topics)')
    if not np.isfinite(values).all() or not np.isfinite(xy[:, :2]).all():
        raise ValueError('Topic weights and coordinates must be finite')
    if np.any(values < 0):
        raise ValueError('Topic weights must be nonnegative')
    return (xy, values, names or [f'Topic{i + 1}' for i in range(values.shape[1])])

def spatial(adata, topic_prop, *, topics=None, size=10, vmax='p99', cmap='viridis', ncols=4, colorbar=True, frameon=False, invert_y=True, invert_x=False, save=None, dpi=300, show=False):
    (xy, values, names) = _topic_data(adata, topic_prop)
    selected = list(range(len(names))) if topics is None else [names.index(t) if isinstance(t, str) else t for t in topics]
    if not selected or ncols < 1 or size <= 0:
        raise ValueError('Select at least one topic; ncols and size must be positive')
    if any((not isinstance(i, (int, np.integer)) or i < 0 or i >= len(names) for i in selected)):
        raise ValueError('Topic index out of range')
    if isinstance(vmax, str):
        if not vmax.startswith('p') or not 0 <= float(vmax[1:]) <= 100:
            raise ValueError("vmax must be numeric, None, or a percentile such as 'p99'")
    ncols = min(ncols, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    (fig, axes) = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False, constrained_layout=True)
    for (ax, idx) in zip(axes.flat, selected):
        weights = values[:, idx]
        order = np.argsort(weights, kind='stable')
        upper = np.percentile(weights, float(vmax[1:])) if isinstance(vmax, str) else vmax
        if upper is not None and (not np.isfinite(upper) or upper < 0):
            plt.close(fig)
            raise ValueError('vmax must be finite and nonnegative')
        points = ax.scatter(xy[order, 0], xy[order, 1], c=weights[order], s=size, cmap=cmap, vmin=0, vmax=upper, edgecolors='none', rasterized=True)
        if invert_y:
            ax.invert_yaxis()
        if invert_x:
            ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.set_title(str(names[idx]))
        if not frameon:
            ax.set_axis_off()
        if colorbar:
            fig.colorbar(points, ax=ax, shrink=0.7, label='Topic proportion')
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor='white')
    if show:
        plt.show()
    return fig

def save_spatial_topics(adata, topic_prop, output_dir, **kwargs):
    (_, values, _) = _topic_data(adata, topic_prop)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(values.shape[1]):
        path = output_dir / f'topic_{i}.png'
        fig = spatial(adata, topic_prop, topics=[i], save=path, **kwargs)
        plt.close(fig)
        paths.append(path)
    return paths


def _finish(fig, save, dpi=180):
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
    return fig


def topic_genes(beta, *, topics=None, top_n=10, ncols=3, save=None):
    from .analysis import validate_beta
    validate_beta(beta)
    selected = list(beta.columns[:6]) if topics is None else list(topics)
    if not selected or top_n < 1 or ncols < 1 or any(t not in beta.columns for t in selected):
        raise ValueError('Select existing topics and positive top_n/ncols')
    ncols = min(ncols, len(selected))
    fig, axes = plt.subplots((len(selected) + ncols - 1) // ncols, ncols, figsize=(5 * ncols, 3.8 * ((len(selected) + ncols - 1) // ncols)), squeeze=False, constrained_layout=True)
    for ax, topic in zip(axes.flat, selected):
        genes = beta[topic].nlargest(top_n).iloc[::-1]
        ax.barh(genes.index.astype(str), genes.values, color='#367a9c')
        ax.set_title(str(topic))
        ax.set_xlabel('Gene weight')
        ax.spines[['top', 'right']].set_visible(False)
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    return _finish(fig, save)


def topic_weights(topic_prop, *, save=None):
    values = np.asarray(topic_prop)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected a finite nonnegative cells-by-topics matrix')
    names = list(topic_prop.columns) if isinstance(topic_prop, pd.DataFrame) else [f'Topic{i + 1}' for i in range(values.shape[1])]
    weights = pd.Series(values.mean(axis=0), index=names).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.25)), constrained_layout=True)
    ax.barh(weights.index.astype(str), weights.values, color='#367a9c')
    ax.set_xlabel('Mean topic proportion across cells')
    ax.spines[['top', 'right']].set_visible(False)
    return _finish(fig, save)


def celltype_spatial(adata, cell_type_key, *, size=1, invert_y=True, invert_x=False, save=None):
    xy, _, _ = _topic_data(adata, np.ones((adata.n_obs, 1)))
    labels = adata.obs[cell_type_key].astype('string').fillna('Unannotated')
    groups = sorted(labels.unique())
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = plt.get_cmap('tab20', len(groups))
    for i, group in enumerate(groups):
        mask = (labels == group).to_numpy(dtype=bool)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=size, color=colors(i), label=group, edgecolors='none', rasterized=True)
    if invert_y:
        ax.invert_yaxis()
    if invert_x:
        ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3, frameon=False, fontsize=8)
    ax.set_title('Original annotations')
    return _finish(fig, save)


def topic_celltype(adata, topic_prop, cell_type_key, *, save=None):
    from .analysis import celltype_activity
    mean, fraction = celltype_activity(adata, topic_prop, cell_type_key)
    x, y = np.meshgrid(np.arange(mean.shape[1]), np.arange(mean.shape[0]))
    fig, ax = plt.subplots(figsize=(max(7, mean.shape[1] * 0.4), max(4, mean.shape[0] * 0.32)), constrained_layout=True)
    dots = ax.scatter(x.ravel(), y.ravel(), c=mean.to_numpy().ravel(), s=fraction.to_numpy().ravel() * 220, cmap='Reds', vmin=0, edgecolors='#555555', linewidths=0.3)
    ax.set_xticks(np.arange(mean.shape[1]))
    ax.set_xticklabels(mean.columns, rotation=90)
    ax.set_yticks(np.arange(mean.shape[0]))
    ax.set_yticklabels(mean.index)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, mean.shape[1] - 0.4)
    ax.set_title('Topic activity within annotated cell types')
    fig.colorbar(dots, ax=ax, shrink=0.6, label='Mean topic proportion')
    handles = [ax.scatter([], [], s=f * 220, c='white', edgecolors='#555555', label=str(int(f * 100)) + '%') for f in [0.1, 0.5, 1.0]]
    ax.legend(handles=handles, title='Dominant-topic fraction', loc='upper left', bbox_to_anchor=(0, -0.18), ncols=3, frameon=False)
    return _finish(fig, save)


def program_markers(adata, topic_prop, selections, *, size=1, invert_y=True, invert_x=False, save=None):
    from scipy.sparse import issparse
    rows = [row for row in selections if row['status'] == 'marker_supported_candidate']
    if not rows:
        raise ValueError('No distinct marker-supported program candidates')
    values = []
    names = []
    hidden = []
    library = np.asarray(adata.X.sum(axis=1)).ravel()
    for row in rows:
        values.append(topic_prop[row['topic']].to_numpy())
        names.append(row['program'] + '\n' + row['topic'])
        for gene in row['matched_markers'][:3]:
            expression = adata[:, [gene]].X
            expression = expression.toarray().ravel() if issparse(expression) else np.asarray(expression).ravel()
            values.append(np.log1p(expression * 10000 / np.maximum(library, 1)))
            names.append(gene + '\nlog1p normalized expression')
        for i in range(3 - min(3, len(row['matched_markers']))):
            hidden.append(len(values))
            values.append(np.zeros(adata.n_obs))
            names.append('Unused' + str(len(values)))
    frame = pd.DataFrame(np.column_stack(values), index=adata.obs_names, columns=names)
    fig = spatial(adata, frame, size=size, ncols=4, colorbar=False, invert_y=invert_y, invert_x=invert_x, dpi=180)
    for i in hidden:
        fig.axes[i].set_visible(False)
    return _finish(fig, save)


def program_activity(adata, topic_prop, selections, cell_type_key, focus_cell_type, *, save=None):
    _topic_data(adata, topic_prop)
    mask = adata.obs[cell_type_key].astype('string').eq(focus_cell_type).fillna(False).to_numpy(dtype=bool)
    rows = [row for row in selections if row['status'] == 'marker_supported_candidate']
    if not mask.any() or len(rows) < 2:
        raise ValueError('Need the focus cell type and two distinct supported programs')
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    series = [topic_prop.loc[mask, row['topic']].to_numpy() for row in rows]
    for i, values in enumerate(series, 1):
        if len(values) > 1 and np.ptp(values) > 0:
            ax.violinplot([values], positions=[i], showmeans=True, showextrema=False)
        else:
            ax.scatter([i], [values.mean()])
    ax.set_xticks(range(1, len(rows) + 1))
    ax.set_xticklabels([row['program'] + '\n' + row['topic'] for row in rows])
    ax.set_ylabel('Topic proportion')
    ax.set_ylim(0, 1)
    ax.set_title('Program activity within ' + focus_cell_type + ' (n=' + str(int(mask.sum())) + ')')
    return _finish(fig, save)
