from graphormer.data import register_dataset
# from dgl.data import QM9
import numpy as np
from sklearn.model_selection import train_test_split

from ..ogb_datasets import PygNodePropPredDataset


@register_dataset("cluster_dataset_1")
def cluster_dataset_1():
    data_dir = "/mnt/zy_data/data/gcn_data/data"
    dataset_name = "languang"
    dataset = PygNodePropPredDataset(dataset_name, data_dir)
    num_graphs = len(dataset)
    print(f"{num_graphs=}")

    split_dict = dataset.split_dict
    train_idx, valid_idx, test_idx = split_dict['train'], split_dict['valid'], split_dict['test']
    # customized dataset split
    # train_valid_idx, test_idx = train_test_split(np.arange(num_graphs), test_size=num_graphs // 10, random_state=0)
    # train_idx, valid_idx = train_test_split(train_valid_idx, test_size=num_graphs // 5, random_state=0)

    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "ogb"
    }
