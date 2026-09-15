import copy
import random

import numpy as np
import torch

from model import NonlinearClassifier


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)


def create_initial_state():
    set_seed()

    model = NonlinearClassifier()

    return copy.deepcopy(model.state_dict())