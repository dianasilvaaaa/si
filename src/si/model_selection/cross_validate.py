
import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset


def K_fold_cross_validation(model: Model, dataset: Dataset, scoring: callable, cv:int, )