from .publaynet import PubLayNetDataset
from .rico import Rico25Dataset
from .scipostlayout import SciPostLayoutDataset

_DATASETS = [
    Rico25Dataset,
    PubLayNetDataset,
    SciPostLayoutDataset
]
DATASETS = {d.name: d for d in _DATASETS}
