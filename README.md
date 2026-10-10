# stOTopic

stOTopic discovers gene programs in spatial transcriptomics data using optimal transport. It learns cell–topic proportions and topic–gene weights, and visualizes their spatial distributions.

## Installation

Requires Python 3.9 or later. Run from the repository root:

```bash
python -m pip install .
```

## Tutorials

- [Lung gene programs](tutorials/lung_gene_programs.ipynb): real-data analysis with example outputs.
- [Quickstart](tutorials/quickstart.ipynb): synthetic data, no download required.
- [Data preparation](tutorials/README.md): input data for the lung tutorial.

Open the notebooks in VS Code or JupyterLab.

## Run the demo

With synthetic data:

```bash
python stOTopic_main.py
```

With your own data:

```bash
python stOTopic_main.py --input data/adata.h5ad --output outputs/my_sample
```

Input is an AnnData `.h5ad` file with raw counts in `X`, unique cell and gene names, and 2D coordinates in `obsm["spatial"]`. Optional annotations can be specified with `--cell-type-key cell_type`.

Run `python stOTopic_main.py --help` for more options.

## Outputs

Results are saved to `outputs/stotopic_demo/` by default:

- `cell_topic.csv.gz`: cell–topic proportions.
- `topic_gene.csv.gz`: gene–topic weights.
- `top_genes.csv` and `topic_summary.csv`: representative genes and topic summaries.
- `figures/`: topic spatial maps, gene weights, and annotation/program plots when available.
- `run_config.json`: run parameters.

The demo presents model outputs and visualizations without evaluation metrics.
