import argparse
import json
from pathlib import Path
import random
import time
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
from . import analysis, stOTopic, pp


def make_demo_data(seed=None):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(15), np.arange(12))
    coords = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    groups = (coords[:, 0] // 5).astype(int)
    rates = np.full((len(coords), 90), 0.3)
    for k in range(3):
        rates[groups == k, k * 30:(k + 1) * 30] = 5.0
    counts = rng.poisson(rates).astype(np.float32)
    data = ad.AnnData(csr_matrix(counts), obs=pd.DataFrame(index=[f'cell_{i}' for i in range(len(coords))]), var=pd.DataFrame(index=[f'gene_{i}' for i in range(90)]))
    data.obsm['spatial'] = coords
    data.obs['synthetic_region'] = pd.Categorical(['Region' + str(i + 1) for i in groups])
    return data


def export_results(adata, model, theta, output_dir, *, cell_type_key=None, focus_cell_type='Tumor', top_n=10, individual_plots=False, flip_x=False, flip_y=True, parameters=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from . import pl
    output = Path(output_dir)
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    names = [f'Topic{i + 1}' for i in range(theta.shape[1])]
    proportions = pd.DataFrame(theta, index=adata.obs_names, columns=names)
    beta = pd.DataFrame(model.get_beta().T, index=model.vocab, columns=names)
    summary = analysis.summarize_topics(proportions, beta, top_n=top_n)
    genes = analysis.top_genes(beta, top_n=top_n)
    if cell_type_key is None:
        cell_type_key = next((key for key in ['cell_type', 'CellType', 'celltype', 'cell_type_annotation', 'synthetic_region'] if key in adata.obs and adata.obs[key].notna().any()), None)
    if cell_type_key is not None and cell_type_key not in adata.obs:
        raise ValueError('Cell-type field not found: ' + cell_type_key)
    selections = analysis.select_programs(beta) if not str(adata.var_names[0]).startswith('gene_') else []
    supported = [row for row in selections if row['status'] == 'marker_supported_candidate']
    selected = list(dict.fromkeys([row['topic'] for row in supported] + summary.sort_values('mean_proportion', ascending=False)['topic'].tolist()))[:6]
    proportions.to_csv(output / 'cell_topic.csv.gz')
    beta.to_csv(output / 'topic_gene.csv.gz')
    genes.to_csv(output / 'top_genes.csv', index=False)
    summary.to_csv(output / 'topic_summary.csv', index=False)
    status = []
    saved = []
    size = 12 if adata.n_obs < 1000 else 1
    def keep(name, fig):
        saved.append('figures/' + name + '.png')
        plt.close(fig)
    keep('topic_genes', pl.topic_genes(beta, topics=selected, top_n=top_n, save=figures / 'topic_genes.png'))
    keep('topic_weights', pl.topic_weights(proportions, save=figures / 'topic_weights.png'))
    keep('topic_spatial', pl.spatial(adata, proportions, size=size, invert_y=not flip_y, invert_x=flip_x, save=figures / 'topic_spatial.png', dpi=180))
    if individual_plots:
        paths = pl.save_spatial_topics(adata, proportions, figures / 'topics', size=size, invert_y=not flip_y, invert_x=flip_x, colorbar=False, dpi=180)
        saved.extend(str(path.relative_to(output)).replace('\\', '/') for path in paths)
    if cell_type_key is not None and adata.obs[cell_type_key].notna().any():
        keep('cell_types', pl.celltype_spatial(adata, cell_type_key, size=size, invert_y=not flip_y, invert_x=flip_x, save=figures / 'cell_types.png'))
    else:
        status.append('Cell-type plots skipped: no usable annotations.')
    if supported:
        keep('program_markers', pl.program_markers(adata, proportions, selections, size=size, invert_y=not flip_y, invert_x=flip_x, save=figures / 'program_markers.png'))
    else:
        status.append('Program-marker plots skipped: no distinct candidates supported by at least two top-20 signature genes.')
    focus_present = cell_type_key is not None and adata.obs[cell_type_key].astype('string').eq(focus_cell_type).fillna(False).any()
    if len(supported) >= 2 and focus_present:
        keep('program_activity', pl.program_activity(adata, proportions, selections, cell_type_key, focus_cell_type, save=figures / 'program_activity.png'))
    else:
        status.append('Within-cell-type program comparison skipped: requires two distinct supported candidates and the requested focus cell type.')
    result = {
        'model': 'stOTopic', 'n_cells': adata.n_obs, 'n_genes': adata.n_vars, 'n_topics': len(names),
        'epochs_run': model.n_iter_, 'cell_type_key': cell_type_key, 'focus_cell_type': focus_cell_type, 'spatial_flip_x': flip_x, 'spatial_flip_y': flip_y,
        'parameters': parameters or {}, 'preprocessing': adata.uns.get('stotopic_preprocessing', {}),
        'program_candidates': selections, 'notes': status, 'figures': saved,
        'interpretation': 'Program names are marker-supported candidates, not validated cell-state assignments. Topic numbers vary across runs.',
    }
    (output / 'run_config.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str), encoding='utf-8')
    report = ['# stOTopic demo', '', 'Cells: ' + str(adata.n_obs) + '; genes: ' + str(adata.n_vars) + '; topics: ' + str(len(names)) + '.', '', '## Model outputs', '', summary.to_string(index=False), '', '## Program candidates', '']
    for row in selections:
        report.append('- ' + row['program'] + ': ' + str(row['topic']) + '; ' + row['status'] + '; markers: ' + ', '.join(row['matched_markers']))
    report.extend(['', result['interpretation'], ''] + status + ['', '## Figures', ''])
    report.extend('![' + Path(path).stem + '](' + path + ')' for path in saved if '/topics/' not in path)
    (output / 'README.md').write_text('\n'.join(report), encoding='utf-8')
    return {'adata': adata, 'theta': proportions, 'beta': beta, 'top_genes': genes, 'topic_summary': summary, 'run_config': result, 'output_dir': output}


def build_parser():
    parser = argparse.ArgumentParser(description='stOTopic gene-program discovery demo')
    parser.add_argument('--input', type=Path, help="Raw-count .h5ad with obsm['spatial']; omitted: synthetic data")
    parser.add_argument('--output', type=Path, default=Path('outputs/stotopic_demo'))
    parser.add_argument('--num-topics', type=int, default=20)
    parser.add_argument('--n-top-genes', type=int, default=6000)
    parser.add_argument('--n-components', type=int, default=400)
    parser.add_argument('--gene-selection', choices=['hvg', 'svg', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--ct-alpha', type=float, default=13.0)
    parser.add_argument('--tg-alpha', type=float, default=1.3)
    parser.add_argument('--theta-temp', type=float, default=0.1)
    parser.add_argument('--early-stop-threshold', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--top-genes', type=int, default=10)
    parser.add_argument('--min-counts', type=int, default=50)
    parser.add_argument('--min-cells', type=int)
    parser.add_argument('--seed', type=int, default=None, help='Optional reproducibility seed; unset by default')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cell-type-key')
    parser.add_argument('--focus-cell-type', default='Tumor')
    sgc = parser.add_mutually_exclusive_group()
    sgc.add_argument('--use-sgc', dest='use_sgc', action='store_true')
    sgc.add_argument('--no-sgc', '--no-use-sgc', dest='use_sgc', action='store_false')
    parser.set_defaults(use_sgc=True)
    parser.add_argument('--sgc-layers', type=int, default=1)
    parser.add_argument('--sgc-mode', choices=['sign', 'sgc'], default='sign')
    parser.add_argument('--individual-plots', action='store_true')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.top_genes < 1:
        parser.error('threads and top-genes must be positive')
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')
    data = ad.read_h5ad(args.input) if args.input else make_demo_data(args.seed)
    if 'spatial' not in data.obsm:
        parser.error("Input requires adata.obsm['spatial'] coordinates")
    if args.cell_type_key is not None and args.cell_type_key not in data.obs:
        parser.error('Cell-type field not found: ' + args.cell_type_key)
    print(f"Input: {args.input or 'synthetic counts'} ({data.n_obs} cells x {data.n_vars} genes)", flush=True)
    data, embeddings = pp.prepare(data, n_top_genes=args.n_top_genes, n_components=args.n_components, gene_selection=args.gene_selection, min_counts=args.min_counts, min_cells=args.min_cells, use_sgc=args.use_sgc, sgc_layers=args.sgc_layers, sgc_mode=args.sgc_mode, random_state=args.seed)
    model = stOTopic(args.num_topics, CT_alpha=args.ct_alpha, TG_alpha=args.tg_alpha, theta_temp=args.theta_temp, device=device)
    started = time.perf_counter()
    theta = model.fit_transform(data, cell_embeddings=embeddings, use_highly_variable=False, epochs=args.epochs, learning_rate=args.learning_rate, early_stop_threshold=args.early_stop_threshold, patience=args.patience)
    parameters = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    parameters.update({'device': device, 'training_seconds': time.perf_counter() - started, 'matmul_precision': torch.get_float32_matmul_precision(), 'allow_tf32': torch.backends.cuda.matmul.allow_tf32})
    result = export_results(data, model, theta, args.output, cell_type_key=args.cell_type_key, focus_cell_type=args.focus_cell_type, top_n=args.top_genes, individual_plots=args.individual_plots, parameters=parameters)
    print(f'Output: theta {result["theta"].shape}; beta {result["beta"].shape}', flush=True)
    print('Saved gene tables, topic plots, and run configuration to ' + str(args.output.resolve()), flush=True)
    for note in result['run_config']['notes']:
        print(note, flush=True)
    return result


if __name__ == '__main__':
    main()
