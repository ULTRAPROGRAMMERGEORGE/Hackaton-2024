from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def eval_mode(module: nn.Module):
    """Контекстный менеджер для того, чтобы делать инференс модели в режиме .eval()
    """
    original_mode = module.training

    try:
        module.eval()
        yield module
    finally:
        module.train(original_mode)
