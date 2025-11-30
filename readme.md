# 交通事故风险预测

## 数据描述


## 成员分工

| 姓名 | 角色 | 分工 |
|------|------|------|
| 金毅阳 | 数据处理 | 清洗与特征工程 |
| 焦祺灏 | 模型设计 | ViT结构优化 |
| 谭清舟 | 模型训练 | 模型训练与调参 |
| 李泽京 | 模型评估 | 设计损失函数与图表绘制 |
| E | 报告撰写 | 汇总成果并撰写报告 |
## 预处理数据说明（preprocessed_data）

下面列出 `preprocessed_data/CHI` 和 `preprocessed_data/NYC` 下的所有 `.npy` 文件、它们的 shape、dtype 以及简要用途（用于训练脚本）。

### CHI
- `new_grid_data_c_4d.npy` — shape=(8784, 7, 10, 10), dtype=float32：时空格点数据张量 (T, D, H, W)。T=时间步数（8784），D=通道数（7），H/W=网格高宽（10×10）。训练时作为动态输入 `all_data_c` 使用，按窗口切片生成样本。
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

如需我也可以把这段同步到项目根目录下的 `readme_preprocessed.md` 或把 `preprocessed_data` 下每个 `.npy` 的数值统计（min/max/mean）也写入说明。

````markdown
**Preprocessed Data (preprocessed_data)**
- **文件**: `preprocessed_data/CHI/new_grid_data_c_4d.npy`: 时空格点数据张量，典型维度为 (T, D, H, W)。含义：T=时间步数；D=通道/特征数；H/W=网格高/宽。该文件供训练脚本作为 `all_data_c` 加载，用于构造时间序列输入 X（按时间窗口切片）。
- **文件**: `preprocessed_data/CHI/new_grid_data_f_4d.npy`: 与上类似，但为另一组目标/通道（在代码中对应 `all_data_f`）。
- **文件**: `preprocessed_data/CHI/new_grid_node_map_c.npy`: 网格到有效节点的映射矩阵，形状通常为 (H*W, valid_node_count)。用于从整网格（包含空白/无用格子）映射到模型使用的有效节点集合，并用于生成节点 mask（有效/无效）。
- **文件**: `preprocessed_data/CHI/new_grid_node_map_f.npy`: 与上类似，对应 `f` 通道的数据集合。
- **文件**: `preprocessed_data/CHI/new_poi_adj_matrix_c.npy`: POI（兴趣点）相似度邻接矩阵，用作语义/结构边矩阵（节点间的语义相似性权重）。训练时可作为图结构输入，增强节点间的语义关系学习。
- **文件**: `preprocessed_data/CHI/new_poi_adj_matrix_f.npy`: 对应 `f` 通道的 POI 相似度矩阵（如果有区分）。
- **文件**: `preprocessed_data/CHI/new_risk_adj_matrix_c.npy`: 风险/事故相似度邻接矩阵（基于历史事故、风险特征计算的节点间权重）。
- **文件**: `preprocessed_data/CHI/new_risk_adj_matrix_f.npy`: 对应 `f` 通道的风险相似度矩阵。
- **文件**: `preprocessed_data/CHI/new_road_adj_matrix_c.npy`: 道路相似度/连通性邻接矩阵（基于道路相近或道路属性计算的节点间权重）。
- **文件**: `preprocessed_data/CHI/new_road_adj_matrix_f.npy`: 对应 `f` 通道的道路邻接矩阵。
- **文件**: `preprocessed_data/CHI/new_static_feat_c.npy`: 节点静态特征矩阵，形状通常为 (valid_node_count, feat_dim)。包含每个节点的常量特征（POI 汇总统计、道路属性、地理/人口等不随时间变化的特征），在模型中与动态时序特征配合使用。
- **文件**: `preprocessed_data/CHI/new_static_feat_f.npy`: 对应 `f` 通道的静态特征文件（若存在区分）。
- **文件**: `preprocessed_data/CHI/ReadMe.txt`: 人类可读的简短说明文件，包含处理步骤摘要与注释（保留以便快速查看）。

- **NYC 子目录**: `preprocessed_data/NYC/` 下包含与 CHI 相同命名和含义的一组文件（针对 NYC 数据集）。

**训练脚本如何使用这些文件**
- 在 `main.py` / `main_ST_Vit_single.py` / `main_ST_Vit_rush.py` 中，`new_grid_data_*_4d.npy` 被加载为时间序列原始数据（变量名如 `all_data_c_path` / `all_data_f_path`），随后通过 `dataloader.py` 中的 `dataset_generate` 做切片与窗口化，转成模型输入 X 和标签 y。
- `new_grid_node_map_*` 被用于构造有效节点 mask 和从格点到节点的索引映射（见训练脚本中对 `grid_node_map` 的求和/mask 操作）。
- 邻接矩阵（`new_poi_adj_matrix_*` / `new_risk_adj_matrix_*` / `new_road_adj_matrix_*`）作为语义/结构边输入到模型（可与 Transformer/图网络模块一并使用）。
- `new_static_feat_*` 与时序特征拼接或并行输入，用于提高静态信息建模能力。

**示例：如何快速加载检查（Python）**
```python
import numpy as np
p = 'preprocessed_data/CHI/new_grid_data_c_4d.npy'
data = np.load(p)
print('data.shape =', data.shape)

node_map = np.load('preprocessed_data/CHI/new_grid_node_map_c.npy')
print('node_map.shape =', node_map.shape)

poi = np.load('preprocessed_data/CHI/new_poi_adj_matrix_c.npy')
print('poi.shape =', poi.shape)

static = np.load('preprocessed_data/CHI/new_static_feat_c.npy')
print('static.shape =', static.shape)
```

**简要处理流程回顾（与生成脚本/笔记本对应）**
- 计算网格中心（见 `data processing/processed/get_grid_centers.py` 或相应 notebook）。
- 将原始流量/事件/POI 数据按网格和时间汇总成 4D 时空张量 `grid_data_*_4d.npy`（T, D, H, W）。
- 生成从格点到有效节点的映射 `grid_node_map_*`（用于压缩/筛选有效节点）。
- 统计并保存节点静态特征 `new_static_feat_*`（POI、道路、天气等静态属性）。
- 构造语义/结构邻接矩阵（POI/risk/road）并保存为 `new_*_adj_matrix_*.npy`。
- 最后把中间产物标准化/格式化并保存到 `preprocessed_data/<CITY>/` 作为训练直接使用的二进制文件（`.npy`）。

如果你希望，我可以：
- 将上述内容直接追加到现有的 `traffic/readme.md`（替换/合并），或者
- 列出 `preprocessed_data/CHI` 与 `NYC` 下每个文件的具体 shape（我可以读取并贴回），或者
- 把 `preprocessed_data` 的生成笔记本单元（step-by-step）列出来以便复现。

````
