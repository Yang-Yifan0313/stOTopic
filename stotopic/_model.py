import torch
from torch import nn
import torch.nn.functional as F
from ._transport import ETP
from ._model_utils import pairwise_euclidean_distance

class stOTopic(nn.Module):

    def __init__(self, num_topics: int, theta_temp: float=1.0, CT_alpha: float=3.0, TG_alpha: float=2.0, use_dsr: bool=True, ct_transport: str='ot', tg_transport: str='ot'):
        super().__init__()
        valid_transports = {'ot', 'softmax'}
        if ct_transport not in valid_transports:
            raise ValueError(f'ct_transport must be one of {sorted(valid_transports)}, got {ct_transport!r}')
        if tg_transport not in valid_transports:
            raise ValueError(f'tg_transport must be one of {sorted(valid_transports)}, got {tg_transport!r}')
        self.num_topics = num_topics
        self.CT_alpha = CT_alpha
        self.TG_alpha = TG_alpha
        self.theta_temp = theta_temp
        self.use_dsr = use_dsr
        self.ct_transport = ct_transport
        self.tg_transport = tg_transport
        self.epsilon = 1e-12

    @staticmethod
    def _softmax_transport(source_embeddings, target_embeddings, alpha):
        distance = pairwise_euclidean_distance(source_embeddings, target_embeddings)
        row_probabilities = F.softmax(-distance * alpha, dim=1)
        return row_probabilities / source_embeddings.shape[0]

    def _cell_topic_transport(self, cell_embeddings):
        if self.ct_transport == 'ot':
            return self.CT_ETP(cell_embeddings, self.topic_embeddings)
        plan = self._softmax_transport(cell_embeddings, self.topic_embeddings, self.CT_alpha)
        return (plan.new_zeros(()), plan)

    def _topic_gene_transport(self):
        if self.tg_transport == 'ot':
            return self.TG_ETP(self.topic_embeddings, self.gene_embeddings)
        plan = self._softmax_transport(self.topic_embeddings, self.gene_embeddings, self.TG_alpha)
        return (plan.new_zeros(()), plan)

    def init(self, vocab_size: int, embed_size: int, _fitted: bool=False, pre_gene_list: list=None, gene_list: list=None, save_init_path: str=None):
        if _fitted:
            topic_embeddings = self.topic_embeddings.data
            assert topic_embeddings.shape == (self.num_topics, embed_size)
            topic_weights = self.topic_weights.data
            del self.topic_weights
        else:
            topic_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty((self.num_topics, embed_size))))
            topic_weights = (torch.ones(self.num_topics) / self.num_topics).unsqueeze(1)
        self.topic_embeddings = nn.Parameter(topic_embeddings)
        self.topic_weights = nn.Parameter(topic_weights)
        gene_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty(vocab_size, embed_size)))
        if _fitted:
            pre_gene_embeddings = self.gene_embeddings.data
            gene_weights = torch.zeros(vocab_size, 1)
            pre_norm_gene_weights = F.softmax(self.gene_weights.data, dim=0)
            del self.gene_embeddings
            del self.gene_weights
            for (i, gene) in enumerate(gene_list):
                if gene in pre_gene_list:
                    pre_gene_idx = pre_gene_list.index(gene)
                    gene_embeddings[i] = pre_gene_embeddings[pre_gene_idx]
                    gene_weights[i] = pre_norm_gene_weights[pre_gene_idx]
            left_avg = (1.0 - gene_weights.sum()) / gene_weights.nonzero().size(0)
            gene_weights[gene_weights == 0] = left_avg
            gene_weights = torch.log(gene_weights)
            gene_weights = gene_weights - gene_weights.mean()
        else:
            gene_weights = (torch.ones(vocab_size) / vocab_size).unsqueeze(1)
        self.gene_embeddings = nn.Parameter(gene_embeddings)
        self.gene_weights = nn.Parameter(gene_weights)
        self.CT_ETP = ETP(self.CT_alpha, init_b_dist=self.topic_weights)
        self.TG_ETP = ETP(self.TG_alpha, init_b_dist=self.gene_weights)
        if save_init_path and (not _fitted):
            import os
            os.makedirs(save_init_path, exist_ok=True)
            torch.save({'topic_embeddings': topic_embeddings.cpu(), 'topic_weights': topic_weights.cpu(), 'gene_embeddings': gene_embeddings.cpu(), 'gene_weights': gene_weights.cpu()}, f'{save_init_path}/init_params.pt')
            print(f'✓ Saved initialization to {save_init_path}/init_params.pt')

    def get_transport_plan_CT(self, cell_embeddings):
        topic_embeddings = self.topic_embeddings.detach().to(cell_embeddings.device)
        if self.ct_transport == 'ot':
            (_, transport_plan) = self.CT_ETP(cell_embeddings, topic_embeddings)
        else:
            transport_plan = self._softmax_transport(cell_embeddings, topic_embeddings, self.CT_alpha)
        return transport_plan.detach().cpu().numpy()

    def get_beta(self):
        with torch.no_grad():
            (_, transport_plan_TG) = self._topic_gene_transport()
            topic_gene_dist = transport_plan_TG * transport_plan_TG.shape[0]
            return topic_gene_dist

    def get_theta(self, cell_embeddings, train_cell_embeddings):
        with torch.no_grad():
            topic_embeddings = self.topic_embeddings.detach().to(cell_embeddings.device)
            dist = pairwise_euclidean_distance(cell_embeddings, topic_embeddings)
            train_dist = pairwise_euclidean_distance(train_cell_embeddings, topic_embeddings)
            log_normalizer = torch.logsumexp(-train_dist / self.theta_temp, dim=0)
            cell_topic_dist = F.softmax(-dist / self.theta_temp - log_normalizer, dim=1)
            return cell_topic_dist

    def forward(self, expression_matrix, cell_embeddings):
        (loss_CT, transport_plan_CT) = self._cell_topic_transport(cell_embeddings)
        (loss_TG, transport_plan_TG) = self._topic_gene_transport()
        loss_ETP = loss_CT + loss_TG
        cell_topic_dist = transport_plan_CT * transport_plan_CT.shape[0]
        topic_gene_dist = transport_plan_TG * transport_plan_TG.shape[0]
        if self.use_dsr:
            reconstruction = torch.matmul(cell_topic_dist, topic_gene_dist)
            loss_DSR = -(expression_matrix * (reconstruction + self.epsilon).log()).sum(axis=1).mean()
        else:
            loss_DSR = loss_ETP.new_zeros(())
        loss = loss_DSR + loss_ETP
        result_dict = {'loss': loss, 'loss_DSR': loss_DSR, 'loss_ETP': loss_ETP, 'loss_CT': loss_CT, 'loss_TG': loss_TG}
        return result_dict
