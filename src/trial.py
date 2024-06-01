import argparse
import numpy as np
import torch
import pygod.generator
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy,load_pokec
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data


dataset = 'region_job'
sens_attr = "region"
predict_attr = "I_am_working_in_field"
label_number =  500
sens_number = 200
seed = 20
path="../dataset/pokec/"
test_idx=False

adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,sens_attr,predict_attr,path=path,label_number=label_number,sens_number=sens_number,seed=seed,test_idx=test_idx)


info = [adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train]
                                                                              
edge_index, edge_attr = from_scipy_sparse_matrix(adj)                                                                                
data = Data(x=features, edge_index = edge_index)

n = int(features.shape[0]*0.05)
k = int(features.shape[0]*0.1)

pygod.generator.gen_contextual_outlier(data, n, k)


# Data 数据类型： https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
# API: https://docs.pygod.org/en/latest/pygod.generator.html#















                                                                                    