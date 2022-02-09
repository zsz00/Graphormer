import shutil, os
import os.path as osp
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset
# from ogb.utils.url import decide_download, download_url, extract_zip
from .dataloader.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from .dataloader.read_graph_raw import extract_zip, read_node_label_hetero, read_nodesplitidx_split_hetero


class PygNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  # original name, e.g., ogbn-proteins

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists. If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0)
            if not self.name in master:
                error_mssg = f'Invalid dataset name {self.name}.\nAvailable datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        # if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
        #     print(self.name + ' has been updated.')
        #     if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
        #         shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  # name of downloaded file, e.g., tox21
        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(PygNodePropPredDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']
        path = osp.join(self.root, 'split', split_type)
        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = torch.from_numpy(train_idx_dict[nodetype]).to(torch.long)
                valid_idx_dict[nodetype] = torch.from_numpy(valid_idx_dict[nodetype]).to(torch.long)
                test_idx_dict[nodetype] = torch.from_numpy(test_idx_dict[nodetype]).to(torch.long)

                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}
        else:
            train_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'train.csv'), header=None).values.T[0]).to(torch.long)
            valid_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'valid.csv'), header=None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(
                pd.read_csv(osp.join(path, 'test.csv'), header=None).values.T[0]).to(torch.long)

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:   # 二进制格式数据文件
            if self.is_hetero:   # 异构图
                return ['edge_index_dict.npz']
            else:
                return ['data.npz']
        else:
            if self.is_hetero:  # 异构图
                return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
            else:
                file_names = ['edge']
                if self.meta_info['has_node_attr'] == 'True':
                    file_names.append('node-feat')
                if self.meta_info['has_edge_attr'] == 'True':
                    file_names.append('edge-feat')
                return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pt')

    def download(self):
        # url = self.meta_info['url']
        # if decide_download(url):
        #     path = download_url(url, self.original_root)
        #     extract_zip(path, self.original_root)
        #     os.unlink(path)
        #     shutil.rmtree(self.root)
        #     shutil.move(osp.join(self.original_root, self.download_name), self.root)
        # else:
        #     print('Stop downloading.')
        #     shutil.rmtree(self.root)
        #     exit(-1)
        pass

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        if self.is_hetero:  # 异构图
            data = read_heterograph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                                        additional_node_files=additional_node_files,
                                        additional_edge_files=additional_edge_files, binary=self.binary)[0]

            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, 'node-label.npz'))
                node_label_dict = {}
                for key in list(tmp.keys()):
                    node_label_dict[key] = tmp[key]
                del tmp
            else:
                node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if 'classification' in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)
                    else:
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.long)
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)

        else:  # 同构图
            # data = pyg_graph_list[0]   data 是一个Data()对象, 可当dict用
            data_list = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files,
                           additional_edge_files=additional_edge_files, binary=self.binary)

        # double-check prediction target   只是检查
        split_dict = self.get_idx_split()
        print(f"{len(data_list)=}, {split_dict['train']=}")
        # assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        # assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        # assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    # pyg_dataset = PygNodePropPredDataset(name='ogbn-mag')
    data_dir = '/mnt/zy_data/data/gcn_data/data'
    # dataset = PygNodePropPredDataset('ogbn-products', data_dir)
    dataset = PygNodePropPredDataset('languang', data_dir)

    print(pyg_dataset[0])
    split_index = pyg_dataset.get_idx_split()
    # print(split_index)
