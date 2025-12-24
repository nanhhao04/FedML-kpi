import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import os


class DataScaler:
    def __init__(self):
        self.scalers = {}
        self.cols_group = {}

    def _identify_columns(self, df):

        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        percent_cols = [c for c in numeric_cols if any(x in c for x in ['UTIL', 'SR', 'EASR', 'RATE', 'FR', 'LOAD'])]

        throughput_cols = [c for c in numeric_cols if 'THROUGHPUT' in c and c not in percent_cols]

        error_cols = [c for c in numeric_cols if any(x in c for x in ['FAIL', 'DROP', 'ERROR', 'DISCARD'])]

        known_cols = set(percent_cols + throughput_cols + error_cols)
        remain_cols = [c for c in numeric_cols if c not in known_cols]

        self.cols_group = {
            'percent': percent_cols,
            'throughput': throughput_cols,
            'error': error_cols,
            'remain': remain_cols
        }
        return numeric_cols

    def fit(self, df):
        """Fit scaler chỉ tren tap train"""
        self._identify_columns(df)

        #  Percent Cols
        if self.cols_group['percent']:
            self.scalers['percent'] = MinMaxScaler()
            self.scalers['percent'].fit(df[self.cols_group['percent']])

        #  Throughput Cols
        if self.cols_group['throughput']:
            self.scalers['throughput'] = StandardScaler()
            self.scalers['throughput'].fit(df[self.cols_group['throughput']])

        #  Error Cols (Log transform trước khi scale)
        if self.cols_group['error']:
            self.scalers['error'] = MinMaxScaler()
            df_log = np.log1p(df[self.cols_group['error']])
            self.scalers['error'].fit(df_log)

        #  Remain Cols
        if self.cols_group['remain']:
            self.scalers['remain'] = RobustScaler()
            self.scalers['remain'].fit(df[self.cols_group['remain']])

    def transform(self, df):
        scaled_df = df.copy()

        if 'percent' in self.scalers:
            cols = self.cols_group['percent']
            scaled_df[cols] = self.scalers['percent'].transform(df[cols])

        if 'throughput' in self.scalers:
            cols = self.cols_group['throughput']
            scaled_df[cols] = self.scalers['throughput'].transform(df[cols])

        if 'error' in self.scalers:
            cols = self.cols_group['error']
            # Log transform trước khi scale
            df_log = np.log1p(df[cols])
            scaled_df[cols] = self.scalers['error'].transform(df_log)

        if 'remain' in self.scalers:
            cols = self.cols_group['remain']
            scaled_df[cols] = self.scalers['remain'].transform(df[cols])

        return scaled_df

def load_and_scale_data(path, node_id):
    start_time = time.time()
    print(f"Reading raw data from {path}...")

    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    # Bỏ cột không cần thiết
    drop_cols = ["NODE_ID", "node", "BEARER_MME_UTIL", "PGW_BEARER_UTIL", "SAU_UTIL_4G",
                 "THROUGHPUT_UTIL", "DEDICATED_BEARER_MME", "NO_PGW_IMS_BEARER", "NO_PGW_SUBS", "Unnamed: 0"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Nội suy dữ liệu thiếu
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()

    elapsed_time = time.time() - start_time
    print(f"[load_and_scale_data] Node {node_id} LOADED & CLEANED (Raw). Shape: {df.shape}")
    # Trả về dữ liệu THÔ chưa scale
    return df


def split_train_val_test(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split dữ liệu THÔ theo thứ tự thời gian.
    """
    n = len(df)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)

    # Split DataFrame
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size: train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    print(f"Split Raw Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def create_sequence_with_date(df, window_len, overlap_rate=0.5):
    step = max(1, int(window_len * (1 - overlap_rate)))
    # Loại bỏ cột date và các cột không phải số để đưa vào numpy
    numeric_df = df.select_dtypes(include=np.number)
    data_np = numeric_df.to_numpy(dtype=np.float32)

    if 'date' in df.columns:
        date_series = df['date'].to_numpy()
    else:
        # Fallback nếu không có date
        date_series = np.arange(len(df))

    total_len = len(df)
    # Tính số lượng window
    if total_len < window_len:
        return np.array([]), pd.DataFrame()

    n_windows = 1 + (total_len - window_len) // step
    num_features = data_np.shape[1]

    X = np.empty((n_windows, window_len, num_features), dtype=np.float32)

    # Metadata để tracking
    meta_data = []

    idx = 0
    for start in range(0, total_len - window_len + 1, step):
        end = start + window_len
        X[idx] = data_np[start:end]

        meta_data.append({
            'start_window': date_series[start],
            'end_window': date_series[end - 1]
        })
        idx += 1

    return X, pd.DataFrame(meta_data)

def process_data_pipeline(raw_df, window_len, overlap_rate=0.5):
    """
    Hàm này kết hợp các bước để đảm bảo KHÔNG Data Leakage:
    1. Split Raw Data.
    2. Fit Scaler chỉ trên Train.
    3. Transform Train, Val, Test.
    4. Xử lý Continuity (nối đuôi) để Val/Test không mất dữ liệu đầu.
    5. Tạo Window (Sequence).
    """
    #  Split
    train_raw, val_raw, test_raw = split_train_val_test(raw_df)

    #  Fit Scaler (Chỉ trên Train)
    scaler = DataScaler()
    scaler.fit(train_raw)

    #  Transform
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    #  Xử lý Continuity & Tạo Sequence
    # Train: Không cần nối đuôi
    X_train, _ = create_sequence_with_date(train_scaled, window_len, overlap_rate)

    # Val: Lấy đuôi của Train nối vào đầu Val để cửa sổ trượt không bị đứt quãng
    # Cần lấy (window_len - step) điểm cuối của train, nhưng an toàn nhất là lấy window_len
    if len(train_scaled) >= window_len:
        # Lấy đoạn cuối train nối với val
        # Lưu ý: Khi nối, index sẽ bị lặp, reset_index là quan trọng
        val_input = pd.concat([train_scaled.iloc[-window_len:], val_scaled]).reset_index(drop=True)
    else:
        val_input = val_scaled

    X_val, _ = create_sequence_with_date(val_input, window_len, overlap_rate)

    # Lấy đuôi của Val nối vào đầu Test
    if len(val_scaled) >= window_len:
        test_input = pd.concat([val_scaled.iloc[-window_len:], test_scaled]).reset_index(drop=True)
    else:
        test_input = test_scaled

    X_test, _ = create_sequence_with_date(test_input, window_len, overlap_rate)

    print(f"Final Sequences: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    return X_train, X_val, X_test


def merge_all_nodes_data(num_clients=3, data_dir="../data", output_file="../data/data_hl19_full.csv"):
    """
    Hàm này giữ nguyên tính năng merge, nhưng output bây giờ là RAW DATA (chưa scale).
    """
    all_dfs = []
    print(f"Bắt đầu hợp nhất dữ liệu từ {num_clients} nodes")

    for i in range(1, num_clients + 1):
        file_path = os.path.join(data_dir, f"../data_hl19_node_{i}.csv")
        if os.path.exists(file_path):
            print(f"Reading: {file_path}")
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        else:
            print(f"Warning: Không tìm thấy file {file_path}")

    if not all_dfs:
        print("Không có dữ liệu nào được load.")
        return None

    merged_df = pd.concat(all_dfs, ignore_index=True)

    if 'date' in merged_df.columns:
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values(by='date')

    merged_df.to_csv(output_file, index=False)
    print(f"Đã lưu file hợp nhất (RAW) tại: {output_file} ({len(merged_df)} dòng)")

    return output_file


if __name__ == '__main__':
    # Test thử luồng
    merge_all_nodes_data()