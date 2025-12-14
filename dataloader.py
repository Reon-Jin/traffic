import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


def is_holiday(date):
    # 2021年美国的节假日列表
    holidays = [
        (1, 1),  # 新年
        (1, 18),  # 马丁·路德·金纪念日
        (2, 15),  # 总统日
        (5, 31),  # 阵亡将士纪念日
        # (7, 4),    # 独立日
        # (7, 5),    # 独立日补假(7月4日是周日)
        # (9, 6),    # 劳动节
        # (10, 11),  # 哥伦布日
        # (11, 11),  # 退伍军人节
        # (11, 25),  # 感恩节
        # (12, 24),  # 圣诞节假期
        # (12, 25),  # 圣诞节
        # (12, 31),  # 新年假期
    ]

    if (date.month, date.day) in holidays:
        return 1
    else:
        return 0


def dataset_generate(all_data_path, grid_node_map, recent_prior=6, week_prior=6,
                     one_day_period=24, days_of_week=7, pre_len=1):
    # 加载数据
    all_data = np.load(all_data_path)[:8736, :, :, :]  # (T,4,H,W)
    D = all_data.shape[1]
    min_val = all_data.min(axis=(0, 2, 3), keepdims=True)
    max_val = all_data.max(axis=(0, 2, 3), keepdims=True)
    normalized_data = (all_data - min_val) / (max_val - min_val + 1e-8)

    # 计算有效样本数
    time_steps, _, h, w = normalized_data.shape
    period = days_of_week * one_day_period
    num_samples = time_steps - recent_prior - week_prior * period - pre_len + 1

    X, y, node_X, time_features = [], [], [], []
    start_date = datetime(2013, 1, 1)

    for i in range(num_samples):
        target_index = i + recent_prior + week_prior * period

        # ================== 生成X_sample的时间步索引 ==================
        # 周历史时间步索引
        current_period_offset = target_index % period
        week_indices = [
            target_index - j * period + current_period_offset
            for j in range(1, week_prior + 1)
        ]
        # 近期历史时间步索引
        recent_indices = list(range(target_index - recent_prior, target_index))
        # 合并所有历史时间步索引
        x_time_indices = week_indices + recent_indices  # 长度=T_prior

        # ================== 生成时间特征（T_prior +1 个） ==================
        # 历史时间步特征（T_prior个）
        x_time_feature = []
        for idx in x_time_indices:
            if idx < 0:
                idx = idx % period  # 循环处理越界
            current_date = start_date + timedelta(hours=idx)
            feature = np.zeros(32)
            feature[current_date.hour] = 1.0
            feature[24 + current_date.weekday()] = 1.0
            feature[31] = float(is_holiday(current_date))
            x_time_feature.append(feature)

        # 目标时间步特征（1个）
        target_date = start_date + timedelta(hours=target_index)
        target_feature = np.zeros(32)
        target_feature[target_date.hour] = 1.0
        target_feature[24 + target_date.weekday()] = 1.0
        target_feature[31] = float(is_holiday(target_date))
        x_time_feature.append(target_feature)  # 合并为 T_prior +1 个

        # 转换为数组
        x_time_feature = np.stack(x_time_feature, axis=0)  # (T_prior+1, 32)

        # ================== 生成X_sample ==================
        # 周历史数据
        week_data = [normalized_data[idx] for idx in week_indices]
        week_data = np.stack(week_data, axis=0)  # (week_prior,4,H,W)

        # 近期历史数据
        recent_data = normalized_data[recent_indices]  # (recent_prior,4,H,W)

        # 合并特征
        X_sample = np.concatenate([week_data, recent_data], axis=0)  # (T_prior,4,H,W)
        assert X_sample.shape == (week_prior + recent_prior, D, h, w)

        # ================== 网格到节点转换 ==================
        X_sample_2d = X_sample.reshape(-1, h * w)
        node_X_sample = (X_sample_2d @ grid_node_map).reshape(X_sample.shape[0], D, -1)
        node_X_sample = node_X_sample[:, 0, :]
        node_X_sample = np.expand_dims(node_X_sample, axis=1)
        # print(node_X_sample.shape)

        # ================== 目标值生成 ==================
        y_sample = normalized_data[target_index: target_index + pre_len, 0, :, :]

        # 存储样本
        X.append(X_sample)
        y.append(y_sample)
        node_X.append(node_X_sample)
        time_features.append(x_time_feature)

    return (
        np.stack(X),  # (num_samples, T_prior, 4, H, W)
        np.stack(y),  # (num_samples, pre_len, H, W)
        np.stack(node_X),  # (num_samples, T_prior, 4, N)
        np.stack(time_features)  # (num_samples, T_prior+1, 32)
    )
