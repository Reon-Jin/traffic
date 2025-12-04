# 交通事故风险预测

## 数据描述


## 成员分工

| 姓名 | 角色 | 分工 |
|------|------|------|
| 金毅阳 | 数据处理 | 清洗与特征工程 |
| 焦祺灏 | 模型设计 | ViT结构优化 |
| 谭清舟 | 模型训练 | 模型训练与调参 |
| 李泽京 | 模型评估 | 设计损失函数与图表绘制 |
| 赵俣鑫 | 报告撰写 | 汇总成果并撰写报告 |
## 预处理数据说明（preprocessed_data）

下面列出 `preprocessed_data/CHI` 和 `preprocessed_data/NYC` 下的所有 `.npy` 文件、它们的 shape、dtype 以及简要用途（用于训练脚本）。

### CHI
- `new_grid_data_c_4d.npy` — shape=(8784, 7, 10, 10), dtype=float32：时空格点数据张量 (T, D, H, W)。T=时间步数（8784），D=通道数（7，包含目标标签`risk_label`,流量信息`inflow`和`outflow`，天气信息`precipitation`、`weather_code`和`wind_speed_10m`。训练时会通过`dataloader.py`文件将7维降为4维，并将相邻一段时期的数据合并为一个样本用于输入模型），H/W=网格高宽（10×10）。训练时作为动态输入 `all_data_c` 使用，按窗口切片生成样本。
- `new_grid_data_f_4d.npy` — shape=(8784, 7, 10, 10), dtype=float32：与上类似，为另一通道集合 `all_data_f`。
- `new_grid_node_map_c.npy` — shape=(100, 63), dtype=int64：网格到有效节点的映射矩阵，行数=H*W=100，列数=valid_node_count=63。用于生成有效节点 mask 和从格点映射到节点索引。
- `new_grid_node_map_f.npy` — shape=(100, 63), dtype=int64：对应 `f` 通道的映射矩阵。
- `new_poi_adj_matrix_c.npy` — shape=(63, 63), dtype=float64：POI 相似度邻接矩阵（语义边），用于图/语义关系建模。
- `new_poi_adj_matrix_f.npy` — shape=(63, 63), dtype=float64：对应 `f` 通道的 POI 矩阵。
- `new_risk_adj_matrix_c.npy` — shape=(63, 63), dtype=float64：风险/事故相似度邻接矩阵。
- `new_risk_adj_matrix_f.npy` — shape=(63, 63), dtype=float64：对应 `f` 通道的风险矩阵。
- `new_road_adj_matrix_c.npy` — shape=(63, 63), dtype=float64：道路相似度/连通性矩阵。
- `new_road_adj_matrix_f.npy` — shape=(63, 63), dtype=float64：对应 `f` 通道的道路矩阵。
- `new_static_feat_c.npy` — shape=(10, 10, 4), dtype=float32：节点静态特征（按网格 H×W 存储，每个格子 4 维静态特征）。
- `new_static_feat_f.npy` — shape=(10, 10, 4), dtype=float32：对应 `f` 通道的静态特征。

### NYC
- `new_grid_data_c_4d.npy` — shape=(8760, 7, 10, 10), dtype=float32：时空数据 (T=8760,... )，注意 NYC 的时间步略有不同。
- `new_grid_data_f_4d.npy` — shape=(8784, 7, 20, 20), dtype=float32：`f` 通道时空数据，网格尺寸为 20×20。
- `new_grid_node_map_c.npy` — shape=(100, 65), dtype=int64：网格→节点映射 (100,65)。
- `new_grid_node_map_f.npy` — shape=(400, 218), dtype=int64：网格→节点映射 (400,218)，对应 20×20 网格的行数 400。
- `new_poi_adj_matrix_c.npy` — shape=(65, 65), dtype=float64：POI 相似度矩阵（65 个有效节点）。
- `new_poi_adj_matrix_f.npy` — shape=(218, 218), dtype=float64：`f` 通道 POI 矩阵（218 节点）。
- `new_risk_adj_matrix_c.npy` — shape=(65, 65), dtype=float64：风险矩阵。
- `new_risk_adj_matrix_f.npy` — shape=(218, 218), dtype=float64：风险矩阵 (`f`)。
- `new_road_adj_matrix_c.npy` — shape=(65, 65), dtype=float64：道路矩阵。
- `new_road_adj_matrix_f.npy` — shape=(218, 218), dtype=float64：道路矩阵 (`f`)。
- `new_static_feat_c.npy` — shape=(10, 10, 4), dtype=float32：静态特征 (10×10×4)。
- `new_static_feat_f.npy` — shape=(20, 20, 4), dtype=float32：静态特征 (20×20×4)。

### 加载示例
```python
import numpy as np
base = 'preprocessed_data/CHI'
print(np.load(base + '/new_grid_data_c_4d.npy').shape)
print(np.load(base + '/new_grid_node_map_c.npy').shape)
print(np.load(base + '/new_poi_adj_matrix_c.npy').shape)
print(np.load(base + '/new_static_feat_c.npy').shape)
```

以上文件在训练脚本中分别被加载为 `all_data_*`、`grid_node_map_*`、`*_adj_matrix_*`、`new_static_feat_*`，由 `dataloader.py` 的 `dataset_generate`/`main_*.py` 使用以构造训练样本和 mask。
