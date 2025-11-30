# preprocess.py
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates


def load_and_scale_data(path, node_id):
    start_time = time.time()

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['node'] = node_id
    df = df.set_index('date')
    # Loại bỏ các trường không cần thiết
    drop_cols = [
        "BEARER_MME_UTIL", "PGW_BEARER_UTIL", "SAU_UTIL_4G",
        "THROUGHPUT_UTIL", "DEDICATED_BEARER_MME", "NO_PGW_IMS_BEARER",
        "NO_PGW_SUBS"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    percent_cols = [c for c in df.columns if 'UTIL' in c or c.endswith('SR') or c.endswith('EASR')]
    throughput_cols = [c for c in df.columns if 'THROUGHPUT' in c and c not in percent_cols]
    count_cols = [c for c in df.columns if ('SAU' in c or 'BEARER' in c or 'SESSION' in c)
                  and c not in percent_cols and c not in throughput_cols]
    error_cols = [c for c in df.columns if ('FAIL' in c or 'DROP' in c or 'ERROR' in c)
                  and c not in percent_cols and c not in throughput_cols and c not in count_cols]



    scaled_df = df.copy()

    if percent_cols:
        scaler_percent = MinMaxScaler()
        scaled_df[percent_cols] = scaler_percent.fit_transform(df[percent_cols])

    if throughput_cols:
        scaler_thr = StandardScaler()
        scaled_df[throughput_cols] = scaler_thr.fit_transform(df[throughput_cols])

    if count_cols:
        scaler_count = RobustScaler()
        scaled_df[count_cols] = scaler_count.fit_transform(df[count_cols])

    if error_cols:
        df[error_cols] = np.log1p(df[error_cols])
        scaler_err = MinMaxScaler()
        scaled_df[error_cols] = scaler_err.fit_transform(df[error_cols])

    scaled_df = scaled_df.ffill().reset_index()

    elapsed_time = time.time() - start_time
    print(f"[load_and_scale_data] Node {node_id} hoàn thành trong {elapsed_time:.2f} giây")

    return scaled_df


def split_train_val_test(X, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    return X_train, X_val, X_test


def create_sequence_with_date(df, window_len, overlap_rate=0.5):
    step = int(window_len * (1 - overlap_rate))
    if step < 1:
        step = 1

    X = []
    output_rows = []

    # Lấy các cột số liệu để tạo chuỗi thời gian (X)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'node' in numeric_cols:
        numeric_cols.remove('node')

    data = df[numeric_cols].values

    # Lặp qua dữ liệu để tạo các cửa sổ
    for start in range(0, len(df) - window_len + 1, step):
        end = start + window_len

        X.append(data[start:end])

        # Lấy hàng cuối cùng của cửa sổ
        end_row = df.iloc[end - 1].copy()

        start_date = df.index[start] if df.index.name == 'date' else df.iloc[start]['date']
        end_date = df.index[end - 1] if df.index.name == 'date' else df.iloc[end - 1]['date']

        end_row['start_window'] = start_date
        end_row['end_window'] = end_date

        # Thêm hàng đã chỉnh sửa vào danh sách
        output_rows.append(end_row)

    X = np.array(X)

    # Tạo DataFrame mới từ danh sách các hàng đã thu thập
    data_with_date = pd.DataFrame(output_rows).reset_index(drop=True)

    return X, data_with_date
