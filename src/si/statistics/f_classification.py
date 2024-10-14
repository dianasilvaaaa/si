

from si.data.dataset import Dataset
import scipy


def f_classification(dataset: Dataset) -> tuple:
    
    classes = dataset.get_classes()

    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.X[ mask , :]
        groups.append(group)

    return scipy.stats.f_oneway(*groups)
