import pytest
import torch
from stotopic._model import stOTopic

@pytest.mark.parametrize('use_dsr,ct_transport,tg_transport', [(True, 'ot', 'ot'), (False, 'ot', 'ot'), (True, 'softmax', 'ot'), (True, 'ot', 'softmax'), (True, 'softmax', 'softmax')])
def test_ablation_losses_are_finite_and_differentiable(use_dsr, ct_transport, tg_transport):
    torch.manual_seed(1)
    model = stOTopic(3, CT_alpha=2.0, TG_alpha=1.0, use_dsr=use_dsr, ct_transport=ct_transport, tg_transport=tg_transport)
    model.init(vocab_size=7, embed_size=4)
    expression = torch.rand(5, 7)
    cells = torch.rand(5, 4)
    output = model(expression, cells)
    assert all((torch.isfinite(value) for value in output.values()))
    output['loss'].backward()
    assert model.topic_embeddings.grad is not None
    assert model.gene_embeddings.grad is not None

def test_without_dsr_excludes_reconstruction_loss():
    model = stOTopic(3, use_dsr=False)
    model.init(vocab_size=7, embed_size=4)
    output = model(torch.rand(5, 7), torch.rand(5, 4))
    assert output['loss_DSR'].item() == 0.0
    assert torch.allclose(output['loss'], output['loss_ETP'])

def test_without_dual_ot_uses_only_dsr_loss():
    model = stOTopic(3, ct_transport='softmax', tg_transport='softmax')
    model.init(vocab_size=7, embed_size=4)
    output = model(torch.rand(5, 7), torch.rand(5, 4))
    assert output['loss_CT'].item() == 0.0
    assert output['loss_TG'].item() == 0.0
    assert torch.allclose(output['loss'], output['loss_DSR'])

def test_softmax_transports_have_expected_row_normalization():
    model = stOTopic(3, ct_transport='softmax', tg_transport='softmax')
    model.init(vocab_size=7, embed_size=4)
    cells = torch.rand(5, 4)
    (_, ct_plan) = model._cell_topic_transport(cells)
    beta = model.get_beta()
    assert torch.allclose(ct_plan.sum(1) * cells.shape[0], torch.ones(5))
    assert torch.allclose(beta.sum(1), torch.ones(3))

def test_invalid_transport_mode_is_rejected():
    with pytest.raises(ValueError, match='ct_transport'):
        stOTopic(3, ct_transport='invalid')
