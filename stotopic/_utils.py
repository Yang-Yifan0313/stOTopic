import numpy as np
import torch
import logging
from typing import List, Callable

class DataIterator:

    def __init__(self, expression_matrix, cell_embeddings, batch_size, device, low_memory: bool=False, shuffle=True):
        self.cell_embeddings = torch.tensor(cell_embeddings, dtype=torch.float32)
        self.low_memory = low_memory
        self.shuffle = shuffle
        self.batch_size = batch_size
        if low_memory:
            self.expression_matrix = expression_matrix
        else:
            self.expression_matrix = torch.tensor(expression_matrix.toarray(), dtype=torch.float32).to(device)
            self.cell_embeddings = self.cell_embeddings.to(device)

    def __iter__(self):
        num_samples = self.expression_matrix.shape[0]
        indices = np.arange(num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, num_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            if self.low_memory:
                batch_expression = torch.tensor(self.expression_matrix[batch_indices].toarray(), dtype=torch.float32)
            else:
                batch_expression = self.expression_matrix[batch_indices]
            batch_cell_embed = self.cell_embeddings[batch_indices]
            yield (batch_expression, batch_cell_embed)

class Dataset:

    def __init__(self, docs: List[str], preprocess: Callable, batch_size: int=200, device: str='cpu', low_memory: bool=False, preset_cell_embeddings=None):
        result = preprocess.preprocess(docs)
        self.expression_matrix = result['train_bow']
        self.gene_list = result['vocab']
        self.vocab_size = len(self.gene_list)
        self.cell_embeddings = preset_cell_embeddings
        self.cell_embed_size = self.cell_embeddings.shape[1]
        self.dataloader = DataIterator(self.expression_matrix, self.cell_embeddings, batch_size, device, low_memory)

def check_fitted(model):
    return model.beta is not None

def assert_fitted(model):
    msg = "This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
    if not check_fitted(model):
        raise ValueError(msg % {'name': type(model).__name__})

class Logger:

    def __init__(self, level):
        self.logger = logging.getLogger('Info')
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f'{message}')

    def warning(self, message):
        self.logger.warning(f'WARNING: {message}')

    def set_level(self, level):
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(sh)
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]
