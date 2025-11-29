import torch
import os
import numpy as np
import dataloader


# 创建数据集对象
class TimeSeriesDataset(Dataset):
    def __init__(self, grid_X_c, grid_y_c, grid_X_f, grid_y_f, node_X_c, node_X_f, target_time_feature):
        self.X_c = torch.tensor(grid_X_c, dtype=torch.float32)
        self.y_c = torch.tensor(grid_y_c, dtype=torch.float32)
        self.X_f = torch.tensor(grid_X_f, dtype=torch.float32)
        self.y_f = torch.tensor(grid_y_f, dtype=torch.float32)
        self.node_X_c = torch.tensor(node_X_c, dtype=torch.float32)
        self.node_X_f = torch.tensor(node_X_f, dtype=torch.float32)
        self.target_time_feature = torch.tensor(target_time_feature, dtype=torch.float32)
        # self.flag_c = torch.tensor(flag_c, dtype=torch.float32)
        # self.flag_f = torch.tensor(flag_f, dtype=torch.float32)

    def __len__(self):
        return len(self.X_c)

    def __getitem__(self, idx):
        return self.X_c[idx], self.y_c[idx], self.X_f[idx], self.y_f[idx], self.node_X_c[idx], self.node_X_f[idx], self.target_time_feature[idx]

# 加载数据
#目录
data_path = os.path.join(os.path.split(curPath)[0], "data",'npy_new_nodiff')
#时空数据：（T,D,H,W）
all_data_c_path = os.path.join(data_path, 'new_grid_data_c_4d.npy')
all_data_f_path = os.path.join(data_path, 'new_grid_data_f_4d.npy')

#映射矩阵：（HW,valid）
grid_node_map_c_path = os.path.join(data_path, 'new_grid_node_map_c.npy')
grid_node_map_f_path = os.path.join(data_path, 'new_grid_node_map_f.npy')
#静态特征
static_feat_c_path = os.path.join(data_path, 'new_static_feat_c.npy')
static_feat_f_path = os.path.join(data_path, 'new_static_feat_f.npy')

#语义边
poi_adj_c_path = os.path.join(data_path, 'new_poi_adj_matrix_c.npy')
poi_adj_f_path = os.path.join(data_path, 'new_poi_adj_matrix_f.npy')
risk_adj_c_path = os.path.join(data_path, 'new_risk_adj_matrix_c.npy')
risk_adj_f_path = os.path.join(data_path, 'new_risk_adj_matrix_f.npy')
road_adj_c_path = os.path.join(data_path, 'new_road_adj_matrix_c.npy')
road_adj_f_path = os.path.join(data_path, 'new_road_adj_matrix_f.npy')


#映射矩阵：（HW,valid）
grid_node_map_c = np.load(grid_node_map_c_path)
grid_node_map_f = np.load(grid_node_map_f_path)
#静态特征
static_feat_c = np.load(static_feat_c_path)
static_feat_f = np.load(static_feat_f_path)

#语义边
poi_adj_c = np.load(poi_adj_c_path)
poi_adj_f = np.load(poi_adj_f_path)
risk_adj_c = np.load(risk_adj_c_path)
risk_adj_f = np.load(risk_adj_f_path)
road_adj_c = np.load(road_adj_c_path)
road_adj_f = np.load(road_adj_f_path)

# 构造数据集
X_c, y_c, node_X_c, target_time_features= dataloader.dataset_generate(all_data_c_path,grid_node_map_c)
X_f, y_f, node_X_f, _= dataloader.dataset_generate(all_data_f_path,grid_node_map_f)

dataset = TimeSeriesDataset(X_c, y_c, X_f, y_f, node_X_c, node_X_f, target_time_features)
