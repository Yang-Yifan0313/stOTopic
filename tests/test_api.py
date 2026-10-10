import json
import sys
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix
from stotopic import stOTopic, metrics, pl, pp
from stotopic.demo import main, make_demo_data
torch.set_num_threads(1)

def test_fit_transform_numpy_inference_and_finetune():
    (data, embeddings) = pp.prepare(make_demo_data(), gene_selection='all', n_components=8)
    original = data.X.copy()
    model = stOTopic(3, device='cpu', theta_temp=0.1)
    theta = model.fit_transform(data, epochs=3, early_stop_threshold=0)
    assert theta.shape == (180, 3)
    assert model.get_beta().shape == (3, 90)
    assert np.isfinite(theta).all()
    np.testing.assert_allclose(theta.sum(axis=1), 1, atol=1e-06)
    np.testing.assert_allclose(theta, model.transform(cell_embeddings=embeddings), atol=1e-06)
    assert (data.X != original).nnz == 0
    model.fit(data, epochs=1)
    with pytest.raises(ValueError, match='gene order'):
        model.fit(data[:, ::-1].copy(), epochs=1)

def test_low_memory_and_invalid_input():
    (data, embeddings) = pp.prepare(make_demo_data(), n_components=5)
    model = stOTopic(3, device='cpu', low_memory=True, low_memory_batch_size=60)
    theta = model.fit_transform(data, epochs=1)
    assert np.isfinite(theta).all()
    with pytest.raises(ValueError, match='cell_embeddings'):
        stOTopic(3).fit(data, cell_embeddings=embeddings[:-1], epochs=1)
    with pytest.raises(ValueError, match='epochs'):
        stOTopic(3).fit(data, epochs=0)
    with pytest.raises(ValueError, match='positive'):
        stOTopic(3, theta_temp=0)
    with pytest.raises(ValueError, match='not fitted'):
        stOTopic(3).transform(cell_embeddings=embeddings)

def test_small_temperature_stays_finite():
    from stotopic._model import stOTopic as Core
    model = Core(3, theta_temp=1e-06)
    model.init(7, 4)
    theta = model.get_theta(torch.randn(5, 4), torch.randn(8, 4))
    assert torch.isfinite(theta).all()
    torch.testing.assert_close(theta.sum(1), torch.ones(5))

def test_metrics_known_values_and_sparse_dense_parity():
    matrix = np.array([[4, 4, 0], [4, 4, 0], [0, 0, 4], [0, 0, 4]], dtype=np.float32)
    data = ad.AnnData(matrix, var=pd.DataFrame(index=['a', 'b', 'c']))
    beta = pd.DataFrame({'Topic1': [3, 2, 0], 'Topic2': [3, 2, 0]}, index=data.var_names)
    assert metrics.get_topic_coherence(data, beta, topk=2) == pytest.approx(1)
    assert metrics.get_topic_diversity(beta, topk=2) == pytest.approx([0.9 ** 2] * 2)
    sparse = data.copy()
    sparse.X = csr_matrix(matrix)
    assert metrics.get_topic_coherence(sparse, beta, topk=2) == metrics.get_topic_coherence(data, beta, topk=2)
    universal = data.copy()
    universal.X = np.ones_like(matrix)
    assert metrics.get_topic_coherence(universal, beta) == 0
    assert metrics.get_topic_diversity(beta.iloc[:, :1]) == [1]

def test_topic_redundancy_overlap_and_validation():
    beta = pd.DataFrame({'A': [4, 3, 2, 1], 'B': [2, 4, 3, 1], 'C': [1, 2, 4, 3]}, index=['a', 'b', 'c', 'd'])
    assert metrics.get_topic_redundancy(beta, topk=2) == pytest.approx(2 / 3)
    assert metrics.get_topic_redundancy(beta[['C', 'A', 'B']], topk=2) == pytest.approx(2 / 3)
    assert metrics.get_topic_redundancy(beta[['A', 'C']], topk=2) == 0
    repeated = pd.concat([beta['A']] * 2, axis=1)
    repeated.columns = ['A', 'B']
    assert metrics.get_topic_redundancy(repeated, topk=2) == 2
    with pytest.raises(ValueError, match='At least two'):
        metrics.get_topic_redundancy(beta[['A']], topk=2)
    with pytest.raises(ValueError, match='exceeds'):
        metrics.get_topic_redundancy(beta, topk=5)
    with pytest.raises(ValueError, match='positive integer'):
        metrics.get_topic_redundancy(beta, topk=0)

def test_spatial_alignment_percentile_orientation_and_no_mutation(tmp_path):
    data = make_demo_data()
    weights = np.random.default_rng(0).random((data.n_obs, 2))
    frame = pd.DataFrame(weights, index=data.obs_names, columns=['Topic1', 'Topic2'])
    before = data.obs.copy()
    fig = pl.spatial(data, frame, topics=['Topic1'], invert_y=False, invert_x=False, save=tmp_path / 'plot.png')
    ax = fig.axes[0]
    assert not ax.yaxis_inverted()
    assert not ax.xaxis_inverted()
    assert ax.collections[0].norm.vmax == pytest.approx(np.percentile(weights[:, 0], 99))
    np.testing.assert_array_equal(ax.collections[0].get_array(), np.sort(weights[:, 0]))
    pd.testing.assert_frame_equal(before, data.obs)
    plt.close(fig)
    paths = pl.save_spatial_topics(data, frame, tmp_path / 'topics')
    assert all((p.stat().st_size > 0 for p in paths))
    with pytest.raises(ValueError, match='index'):
        pl.spatial(data, frame.iloc[::-1])

def test_prepare_copies_counts_and_hvg():
    data = make_demo_data()
    (selected, embeddings) = pp.prepare(data, n_top_genes=40, n_components=8)
    assert selected.shape == (180, 40)
    assert data.shape == (180, 90)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-06)
    expected = data[:, selected.var_names].X
    assert (selected.X != expected).nnz == 0
    assert 'log1p' not in selected.uns
    with pytest.raises(ValueError, match='raw counts'):
        bad = data.copy()
        bad.uns['log1p'] = {}
        pp.prepare(bad)

@pytest.mark.parametrize('sgc_args, expected_sgc', [([], True), (['--no-sgc'], False), (['--use-sgc'], True)])
def test_demo_outputs_and_h5ad_input(tmp_path, sgc_args, expected_sgc):
    source = tmp_path / 'input.h5ad'
    make_demo_data().write_h5ad(source)
    main(['--input', str(source), '--output', str(tmp_path / 'result'), '--device', 'cpu', '--epochs', '2', '--n-components', '8'] + sgc_args)
    result = json.loads((tmp_path / 'result/run_config.json').read_text())
    assert result['parameters']['use_sgc'] is expected_sgc
    assert result['model'] == 'stOTopic'
    assert result['n_cells'] == 180
    assert result['parameters']['sgc_mode'] == 'sign'
    assert result['parameters']['seed'] is None
    assert result['parameters']['ct_alpha'] == 13.0
    assert not (tmp_path / 'result/metrics.json').exists()
    assert (tmp_path / 'result/figures/topic_spatial.png').stat().st_size > 0
    assert 'figures/topic_celltype.png' not in result['figures']
    assert not (tmp_path / 'result/figures/topic_celltype.png').exists()
    assert result['spatial_flip_x'] is False
    assert result['spatial_flip_y'] is True
    assert pd.read_csv(tmp_path / 'result/cell_topic.csv.gz', index_col=0).shape == (180, 20)
    assert 'sctm' not in sys.modules

@pytest.mark.parametrize("sparse_format, expected_stride", [("csc", (1, 180)), ("csr", (90, 1))])
def test_training_preserves_input_sparse_layout(sparse_format, expected_stride):
    data = make_demo_data()
    data.X = data.X.asformat(sparse_format)
    data, embeddings = pp.prepare(data, gene_selection="all", n_components=5, use_sgc=False)
    model = stOTopic(3, device="cpu")
    observed = []
    handle = model.model.register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0].stride()))
    model.fit_transform(data, cell_embeddings=embeddings, use_highly_variable=False, epochs=1)
    handle.remove()
    assert observed == [expected_stride]
