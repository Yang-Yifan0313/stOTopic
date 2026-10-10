import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
from stotopic import analysis, pl
from stotopic.demo import make_demo_data, build_parser


def test_celltype_activity_uses_within_group_denominators_and_skips_missing():
    data = make_demo_data(1)[:4].copy()
    data.obs['label'] = ['A', 'A', 'B', None]
    theta = pd.DataFrame([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [1, 0]], index=data.obs_names, columns=['T1', 'T2'])
    mean, fraction = analysis.celltype_activity(data, theta, 'label')
    np.testing.assert_allclose(mean.loc['A'], [0.65, 0.35])
    np.testing.assert_allclose(fraction.loc['A'], [0.5, 0.5])
    np.testing.assert_allclose(fraction.loc['B'], [0, 1])
    assert list(mean.index) == ['A', 'B']
    fig = pl.topic_celltype(data, theta, 'label')
    plt.close(fig)
    with pytest.raises(ValueError, match='index'):
        analysis.celltype_activity(data, theta.iloc[::-1], 'label')


def test_program_selection_does_not_invent_distinct_states_or_missing_markers():
    beta = pd.DataFrame({'T1': [5, 4, 3, 2, 0], 'T2': [0, 0, 0, 0, 9]}, index=['a', 'b', 'c', 'd', 'e'])
    signatures = {'A': ['a', 'b'], 'B': ['c', 'd'], 'C': ['missing']}
    selected = analysis.select_programs(beta, signatures, top_n=4)
    assert selected[0]['status'] == 'shared_topic_no_separate_programs'
    assert selected[1]['status'] == 'shared_topic_no_separate_programs'
    assert selected[2]['topic'] is None
    assert selected[2]['available_markers'] == []


def test_summary_and_gene_rankings_preserve_topic_identity():
    theta = pd.DataFrame([[0.8, 0.2], [0.1, 0.9]], columns=['B', 'A'])
    beta = pd.DataFrame({'B': [0.1, 0.9], 'A': [0.8, 0.2]}, index=['g1', 'g2'])
    summary = analysis.summarize_topics(theta, beta, top_n=1)
    assert summary['top_genes'].tolist() == ['g2', 'g1']
    assert summary['dominant_cells'].tolist() == [1, 1]
    genes = analysis.top_genes(beta, top_n=1)
    assert genes['gene'].tolist() == ['g2', 'g1']
    with pytest.raises(ValueError, match='columns'):
        analysis.summarize_topics(theta[['A', 'B']], beta)


def test_demo_default_configuration_and_unseeded_data():
    args = build_parser().parse_args([])
    assert (args.num_topics, args.n_components, args.epochs) == (20, 400, 10000)
    assert (args.ct_alpha, args.tg_alpha, args.learning_rate, args.theta_temp) == (13.0, 1.3, 0.01, 0.1)
    assert args.use_sgc and args.sgc_mode == 'sign' and args.seed is None
    assert (make_demo_data().X != make_demo_data().X).nnz > 0
