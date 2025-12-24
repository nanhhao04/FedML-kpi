import pandas as pd
import time
import torch
import numpy as np
import os
import yaml
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMModel, AELSTM, VAELSTM, VAE, AECNN
from preprocess import load_and_scale_data, create_sequence_with_date, process_data_pipeline
from fedml_logic import compute_loss

cfg = yaml.safe_load(open("config.yml", "r"))


# Trung bình 10 round cuối eval trong FL ( Không dùng nữa )
n_tail = 10
def get_metric_tail(data_path, n_tail = 10):
    df = pd.read_csv(data_path)
    df_last_n = df.tail(n_tail)
    #avg_loss = df_last_n['loss'].mean()
    avg_mae = df_last_n['mae'].mean()
    avg_mse = df_last_n['mse'].mean()
    ##print(f"Metric: MAE: {avg_mae:.4f} | MSE: {avg_mse:.4f}")

    return avg_mae, avg_mse

# Trung bình 10 round khi inference ( Sau FL )
def get_metric_eval(model_path, data_path, cfg, n_rounds=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_and_scale_data(data_path, node_id = None)
    window_len = cfg["model"]["window_len"]
    overlap_rate = cfg["model"]["overlap"]
    _, _, X_test = process_data_pipeline(df, window_len, overlap_rate)

    input_dim = X_test.shape[-1]
    print(f"Data Input Dim: {input_dim}")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model_type = cfg["model"]["type"]
    if model_type == "lstm":
        model = LSTMModel(input_dim=input_dim)
    elif model_type == "ae_lstm":
        model = AELSTM(input_dim=input_dim)
    elif model_type == "vae_lstm":
        model = VAELSTM(input_dim=input_dim)
    elif model_type == "vae":
        model = VAE(input_dim=input_dim * window_len)
    elif model_type == "ae_cnn":
        model = AECNN(input_dim=input_dim, seq_len=window_len)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)


    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    print("Đã load weights thành công.")

    # Eval model
    model.eval()
    accum_mse = 0.0
    accum_mae = 0.0


    with torch.no_grad():
        for i in range(n_rounds):
            round_mse = 0.0
            round_mae = 0.0
            total_samples = 0
            for batch in test_loader:
                xb = batch[0].to(device)

                _,mse,mae,_ = compute_loss(model, xb)
                if isinstance(mse, torch.Tensor): mse = mse.item()
                if isinstance(mae, torch.Tensor): mae = mae.item()

                bs = xb.size(0)
                round_mse += mse * bs
                round_mae += mae * bs
                total_samples += bs

                # Trung bình của vòng hiện tại
            avg_round_mse = round_mse / total_samples
            avg_round_mae = round_mae / total_samples

            accum_mse += avg_round_mse
            accum_mae += avg_round_mae

        final_mse = accum_mse / n_rounds
        final_mae = accum_mae / n_rounds

    return final_mse, final_mae


def get_inference_data(csv_path, seq_len=6):
    print(f"Reading and processing {csv_path} using preprocess.py logic...")

    # TIME: LOAD + SCALE
    t0 = time.time()
    try:
        df_scaled = load_and_scale_data(csv_path, node_id=0)
    except Exception as e:
        print(f"Lỗi khi gọi load_and_scale_data: {e}")
        df_scaled = pd.read_csv(csv_path)
        if "date" in df_scaled.columns:
            df_scaled["date"] = pd.to_datetime(df_scaled["date"])
    t1 = time.time()
    print(f"[TIME] load_and_scale_data: {(t1 - t0):.4f}s")

    # TIME: CREATE SEQUENCES
    print(f"Creating sequences with length {seq_len}...")
    t0 = time.time()
    X, _ = create_sequence_with_date(df_scaled, window_len=seq_len, overlap_rate=0.5)
    t1 = time.time()
    print(f"[TIME] create_sequence_with_date: {(t1 - t0):.4f}s")

    if len(X) == 0:
        raise ValueError("Không tạo được sequence nào.")

    #  TIME: CONVERT TO TENSOR
    t0 = time.time()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    t1 = time.time()
    print(f"[TIME] convert to tensor: {(t1 - t0):.4f}s")

    return X_tensor


def count_parameters(model):
    """Đếm số lượng parameters của model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference(csv_path, seq_len=6, device="cpu"):
    # PREPARE DATA
    t0_total = time.time()

    X_test = get_inference_data(csv_path, seq_len)
    X_test = X_test.to(device)

    num_samples = X_test.shape[0]
    input_dim = X_test.shape[2]

    print(f"\n{'=' * 70}")
    print(f" BENCHMARK INFERENCE TIME (Using preprocess logic)")
    print(f" File: {csv_path}")
    print(f" Input Shape: {X_test.shape}")
    print(f" Device: {device.upper()}")
    print(f"{'=' * 70}")
    print(f"{'Model Name':<25} | {'Time (s)':<12} | {'ms/sample':<12} | Params")
    print(f"{'-' * 70}")

    models_to_test = {
        "LSTM (Forecasting)": LSTMModel(input_dim=input_dim, hidden_dim=64, layers=2),
        "AE-LSTM": AELSTM(input_dim=input_dim, enc_hidden=64, dec_hidden=64),
        "VAE-LSTM": VAELSTM(input_dim=input_dim, seq_len=seq_len, hidden_dim=64),
        "VAE (Standard)": VAE(input_dim=input_dim, hidden_dim=64),
        "AE-CNN": AECNN(input_dim=input_dim, seq_len=seq_len, hidden_channels=[32, 16]),
    }

    results = {}

    # BENCHMARK
    for name, model in models_to_test.items():
        model.to(device)
        model.eval()

        # Warmup
        try:
            t0 = time.time()
            with torch.no_grad():
                _ = model(X_test[:1])
            t1 = time.time()
            warmup_time = t1 - t0
        except Exception as e:
            print(f"[WARN] Warmup failed for {name}: {e}")
            continue

        # Inference timing
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            _ = model(X_test[:16])
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        total_time = t1 - t0
        ms_per_sample = (total_time / 16) * 1000
        n_params = count_parameters(model)

        print(f"{name:<25} | {total_time:.5f}     | {ms_per_sample:.3f} ms      | {n_params:,}")

        results[name] = {
            "time": total_time,
            "ms/sample": ms_per_sample,
            "params": n_params,
            "warmup": warmup_time,
        }
    print(f"{'=' * 70}")
    print(f" Tổng thời gian chạy toàn pipeline: {time.time() - t0_total:.4f} s")
    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    TEST_FILE = "../data/data_hl19_full.csv"
    benchmark_inference(csv_path=TEST_FILE, seq_len=6, device="cpu")
