import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix, issparse
import scanpy as sc
from typing import Optional
from ._model import stOTopic as _stOTopicModel
from ._utils import Logger, assert_fitted, check_fitted
logger = Logger('WARNING')

class SingleCellPreprocessor:

    def __init__(self, max_genes=None, verbose=False):
        self.max_genes = max_genes
        self.verbose = verbose
        self.gene_list = None
        self._expression_matrix = None
        self._gene_names = None

    def setup(self, expression_matrix, gene_names):
        if self.max_genes and len(gene_names) > self.max_genes:
            gene_sums = np.array(expression_matrix.sum(axis=0)).flatten()
            top_indices = np.argsort(gene_sums)[-self.max_genes:]
            expression_matrix = expression_matrix[:, top_indices]
            gene_names = [gene_names[i] for i in top_indices]
        self._expression_matrix = expression_matrix
        self._gene_names = gene_names
        self.gene_list = gene_names

    def preprocess(self, cells):
        if self._expression_matrix is None:
            raise ValueError('Must call setup() first with expression matrix data')
        if not issparse(self._expression_matrix):
            self._expression_matrix = csr_matrix(self._expression_matrix)
        return {'train_bow': self._expression_matrix, 'vocab': self.gene_list}

class SingleCellDataset:

    def __init__(self, expression_matrix, gene_names, cell_embeddings, batch_size, device='cpu', low_memory=False):
        self.expression_matrix = expression_matrix
        self.gene_names = gene_names
        self.cell_embeddings = cell_embeddings
        self.batch_size = batch_size
        self.device = device
        self.low_memory = low_memory
        self._expression_tensor = None
        self._embedding_tensor = None
        if not self.low_memory:
            dense_expression = (self.expression_matrix.toarray() if issparse(self.expression_matrix) else np.asarray(self.expression_matrix)).astype(np.float32, copy=False)
            self._expression_tensor = torch.from_numpy(dense_expression).to(self.device)
            self._embedding_tensor = torch.as_tensor(self.cell_embeddings, dtype=torch.float32, device=self.device)
        if not issparse(self.expression_matrix):
            self.expression_matrix = csr_matrix(self.expression_matrix)
        self.n_cells = expression_matrix.shape[0]
        self.n_genes = expression_matrix.shape[1]
        self.embedding_dim = cell_embeddings.shape[1]
        self.gene_list = gene_names

    @property
    def vocab_size(self):
        return self.n_genes

    @property
    def doc_embed_size(self):
        return self.embedding_dim

    @property
    def vocab(self):
        return self.gene_list

    @property
    def doc_embeddings(self):
        return self.cell_embeddings

    @property
    def dataloader(self):
        n_batches = (self.n_cells + self.batch_size - 1) // self.batch_size
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_cells)
            if not self.low_memory:
                yield (self._expression_tensor[start_idx:end_idx], self._embedding_tensor[start_idx:end_idx])
                continue
            batch_expression = self.expression_matrix[start_idx:end_idx].toarray()
            batch_expression = torch.FloatTensor(batch_expression)
            batch_embeddings = self.cell_embeddings[start_idx:end_idx]
            batch_embeddings = torch.FloatTensor(batch_embeddings)
            if not self.low_memory:
                batch_expression = batch_expression.to(self.device)
                batch_embeddings = batch_embeddings.to(self.device)
            yield (batch_expression, batch_embeddings)

class stOTopic:

    def __init__(self, num_topics: int, num_top_genes: int=15, device: str=None, CT_alpha: float=3.0, TG_alpha: float=2.0, theta_temp: float=1.0, use_dsr: bool=True, ct_transport: str='ot', tg_transport: str='ot', low_memory: bool=False, low_memory_batch_size: int=None, verbose: bool=False, log_interval: int=10, save_init_path: str=None):
        if not isinstance(num_topics, int) or num_topics < 1:
            raise ValueError('num_topics must be a positive integer')
        if any((not np.isfinite(v) or v <= 0 for v in (CT_alpha, TG_alpha, theta_temp))):
            raise ValueError('CT_alpha, TG_alpha and theta_temp must be finite and positive')
        if log_interval < 1 or num_top_genes < 1:
            raise ValueError('log_interval and num_top_genes must be positive')
        if low_memory and (low_memory_batch_size is None or low_memory_batch_size < 1):
            raise ValueError('low_memory requires a positive low_memory_batch_size')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_topics = num_topics
        self.num_top_genes = num_top_genes
        self.low_memory = low_memory
        self.low_memory_batch_size = low_memory_batch_size
        self.topic_gene_dist = None
        self.cell_topic_dist = None
        self.model = _stOTopicModel(num_topics, theta_temp, CT_alpha, TG_alpha, use_dsr=use_dsr, ct_transport=ct_transport, tg_transport=tg_transport)
        self.save_init_path = save_init_path
        self.preprocessor = None
        self.gene_list = None
        self._expression_matrix = None
        self._gene_names = None
        self.log_interval = log_interval
        self.verbose = verbose
        if verbose:
            logger.set_level('DEBUG')
        else:
            logger.set_level('WARNING')
        logger.info(f'use device: {device}')

    def make_optimizer(self, learning_rate: float):
        args_dict = {'params': self.model.parameters(), 'lr': learning_rate}
        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def fit(self, adata: sc.AnnData, cell_embeddings: Optional[np.ndarray]=None, embedding_key: str='X_pca', use_highly_variable: bool=True, epochs: int=200, learning_rate: float=0.002, early_stop_threshold: float=0.1, patience: int=5):
        self.fit_transform(adata=adata, cell_embeddings=cell_embeddings, embedding_key=embedding_key, use_highly_variable=use_highly_variable, epochs=epochs, learning_rate=learning_rate, early_stop_threshold=early_stop_threshold, patience=patience)
        return self

    def fit_transform(self, adata: sc.AnnData, cell_embeddings: Optional[np.ndarray]=None, embedding_key: str='X_pca', use_highly_variable: bool=True, epochs: int=200, learning_rate: float=0.002, early_stop_threshold: float=0.1, patience: int=5):
        if self.verbose:
            logger.info('Starting stOTopic training')
        if cell_embeddings is None:
            if embedding_key not in adata.obsm:
                raise ValueError(f"Cell embedding key '{embedding_key}' not found in adata.obsm")
            cell_embeddings = adata.obsm[embedding_key]
            if self.verbose:
                logger.info(f"Loaded cell embeddings from adata.obsm['{embedding_key}']")
        cell_embeddings = np.asarray(cell_embeddings, dtype=np.float32)
        if cell_embeddings.ndim != 2 or cell_embeddings.shape[0] != adata.n_obs or cell_embeddings.shape[1] < 1 or (not np.isfinite(cell_embeddings).all()):
            raise ValueError('cell_embeddings must be finite with shape (n_cells, n_features)')
        if epochs < 1 or patience < 1 or (not np.isfinite(learning_rate)) or (learning_rate <= 0):
            raise ValueError('epochs, patience and learning_rate must be positive')
        if not np.isfinite(early_stop_threshold) or early_stop_threshold < 0:
            raise ValueError('early_stop_threshold must be finite and nonnegative')
        if not adata.obs_names.is_unique or not adata.var_names.is_unique:
            raise ValueError('Observation and gene names must be unique')
        if self.verbose:
            logger.info(f'Cell embeddings shape: {cell_embeddings.shape}')
        if use_highly_variable and 'highly_variable' in adata.var.columns:
            adata_subset = adata[:, adata.var.highly_variable].copy()
            if self.verbose:
                logger.info(f'Using highly variable genes: {adata_subset.n_vars} genes')
        else:
            adata_subset = adata.copy()
            if self.verbose:
                logger.info(f'Using all genes: {adata_subset.n_vars} genes')
        if issparse(adata_subset.X):
            expression_matrix = adata_subset.X
        else:
            expression_matrix = csr_matrix(adata_subset.X)
        expression_matrix = expression_matrix.astype(np.float32)
        if min(expression_matrix.shape) == 0:
            raise ValueError('Training requires at least one cell and gene')
        if not np.isfinite(expression_matrix.data).all() or (expression_matrix.data < 0).any():
            raise ValueError('Expression values must be finite and nonnegative')
        if (np.asarray(expression_matrix.sum(axis=1)).ravel() == 0).any():
            raise ValueError('Remove cells with zero total counts before training')
        gene_names = adata_subset.var_names.tolist()
        if check_fitted(self):
            if gene_names != self.gene_list or cell_embeddings.shape[1] != self.model.topic_embeddings.shape[1]:
                raise ValueError('Fine-tuning requires the same gene order and embedding dimensions')
        if self.verbose:
            logger.info(f'Expression matrix shape: {expression_matrix.shape}')
            logger.info(f'Number of cells: {adata_subset.n_obs}')
            logger.info(f'Number of genes: {adata_subset.n_vars}')
        self._expression_matrix = expression_matrix
        self._gene_names = gene_names
        self.gene_list = gene_names
        n_cells = expression_matrix.shape[0]
        if self.low_memory:
            logger.info('Using low memory mode')
            assert self.low_memory_batch_size is not None
            batch_size = self.low_memory_batch_size
            dataset_device = 'cpu'
        else:
            batch_size = n_cells
            dataset_device = self.device
        _fitted = check_fitted(self)
        if _fitted:
            logger.info('Fine-tuning the model')
        else:
            logger.info('First fit the model')
        dataset = SingleCellDataset(expression_matrix=expression_matrix, gene_names=gene_names, cell_embeddings=cell_embeddings, batch_size=batch_size, device=dataset_device, low_memory=self.low_memory)
        self.train_cell_embeddings = torch.as_tensor(dataset.doc_embeddings, dtype=torch.float32)
        if not self.low_memory:
            self.train_cell_embeddings = self.train_cell_embeddings.to(self.device)
        n_genes = dataset.vocab_size
        embedding_dim = dataset.doc_embed_size
        if not _fitted:
            self.model.init(n_genes, embedding_dim, save_init_path=self.save_init_path)
        self.gene_list = dataset.vocab
        self.model = self.model.to(self.device)
        optimizer = self.make_optimizer(learning_rate)
        self.model.train()
        self.loss_history_ = []
        self.n_iter_ = 0
        prev_loss = float('inf')
        patience_counter = 0
        for epoch in tqdm(range(1, epochs + 1), desc='Training model', disable=not self.verbose):
            loss_dict = defaultdict(float)
            for (batch_expression, batch_embeddings) in dataset.dataloader:
                if self.low_memory:
                    batch_embeddings = batch_embeddings.to(self.device)
                    batch_expression = batch_expression.to(self.device)
                output_dict = self.model(batch_expression, batch_embeddings)
                batch_loss = output_dict['loss']
                if not torch.isfinite(batch_loss):
                    raise FloatingPointError('Nonfinite training loss; check input scale and hyperparameters')
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                for key in output_dict:
                    loss_dict[key] += output_dict[key].item() * batch_expression.shape[0]
                if self.low_memory:
                    del batch_expression, batch_embeddings, batch_loss, output_dict
                    torch.cuda.empty_cache()
            current_loss = loss_dict['loss'] / n_cells
            self.loss_history_.append(current_loss)
            self.n_iter_ = epoch
            loss_change = abs(current_loss - prev_loss)
            if epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_dict:
                    output_log += f' {key}: {loss_dict[key] / n_cells:.3f}'
                output_log += f' loss_change: {loss_change:.4f}'
                logger.info(output_log)
            if loss_change < early_stop_threshold:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping at epoch {epoch}: loss change {loss_change:.4f} < {early_stop_threshold}')
                    break
            else:
                patience_counter = 0
            prev_loss = current_loss
        self.topic_gene_dist = self.get_topic_gene_dist()
        self.cell_topic_dist = self.transform(cell_embeddings=self.train_cell_embeddings)
        return self.cell_topic_dist

    def transform(self, adata: sc.AnnData=None, cell_embeddings: np.ndarray=None):
        assert_fitted(self)
        if cell_embeddings is None:
            if adata is None or 'X_pca' not in adata.obsm:
                raise ValueError("Provide cell_embeddings or adata.obsm['X_pca']")
            cell_embeddings = adata.obsm['X_pca']
        cell_embeddings = torch.as_tensor(cell_embeddings, dtype=torch.float32, device=self.device)
        if cell_embeddings.ndim != 2 or cell_embeddings.shape[1] != self.model.topic_embeddings.shape[1] or (not torch.isfinite(cell_embeddings).all()):
            raise ValueError('Invalid embedding shape or nonfinite values')
        reference = self.train_cell_embeddings.to(self.device)
        with torch.no_grad():
            self.model.eval()
            theta = self.model.get_theta(cell_embeddings, reference)
            theta = theta.detach().cpu().numpy()
        return theta

    def get_topic_gene_dist(self):
        if not hasattr(self.model, 'gene_embeddings'):
            raise ValueError('Call fit before requesting topic-gene weights')
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def get_beta(self):
        return self.get_topic_gene_dist()

    @property
    def topic_embeddings(self):
        return self.model.topic_embeddings.detach().cpu().numpy()

    @property
    def gene_embeddings(self):
        return self.model.gene_embeddings.detach().cpu().numpy()

    @property
    def vocab(self):
        return self.gene_list

    @property
    def preprocess(self):
        if self.preprocessor is None:
            self.preprocessor = SingleCellPreprocessor(verbose=self.verbose)
        return self.preprocessor

    @property
    def beta(self):
        return self.topic_gene_dist

    @property
    def train_theta(self):
        return self.cell_topic_dist
