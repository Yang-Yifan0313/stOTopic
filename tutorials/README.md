# stOTopic tutorials

- `quickstart.ipynb`: synthetic counts and synthetic region labels; no download.
- `lung_gene_programs.ipynb`: real Lung9 Rep1 analysis with stored example outputs.

Install the package from the repository root and install a notebook frontend:

```bash
python -m pip install .
python -m pip install jupyterlab
jupyter lab
```

## Real-data setup

The real-data notebook expects `data/Lung9_Rep1.h5ad` relative to the repository
root. Alternatively set the `STOTOPIC_DEMO_INPUT` environment variable to your
AnnData file or change `input_path` in the first cell.

This is Lung9 Rep1 from the public CosMx SMI NSCLC dataset. Original data are
available through the [CosMx NSCLC dataset page](https://nanostring.com/products/cosmx-spatial-molecular-imager/nsclc-ffpe-dataset/).

The demonstrated input contains 87,606 cells and 960 panel genes before filtering.
The package expects original integer counts in `X`, unique cell/gene identifiers,
2D coordinates in `obsm['spatial']`, and gene symbols in `var_names`.
For the cell-type panels, `obs['cell_type']` contains original annotations,
including `Tumor`. Labels are used only after fitting. Adapt the field name or
skip annotation panels for other inputs. Data are not bundled in the release.

The tutorial exports full matrices, ranked genes, topic summaries, spatial plots and
configuration under `outputs/lung_tutorial`. The synthetic tutorial uses
`outputs/quickstart`. 

`topic_gene.csv.gz` is genes by topics for plotting. All exported matrices retain cell/gene identifiers.
