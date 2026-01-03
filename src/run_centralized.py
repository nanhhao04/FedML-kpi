
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

from torch.utils.data import DataLoader, TensorDataset

from fedml_logic import FedLocalTrain, compute_loss
from models import VAELSTM, VAE, AELSTM, AECNN, LSTMWithIForest, LSTMModel
from utils import get_metric_tail, get_metric_eval
import time
import argparse
import wandb
from preprocess import load_and_scale_data, process_data_pipeline

# Load config
cfg = yaml.safe_load(open("config.yml", "r"))
FULL_DATA_PATH = "../data/data_hl19_full.csv"
DATA_PATH_REAL = "../data/data_hl19_real.csv"




OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = f"{OUTPUT_DIR}/centralized_metrics.csv"


def train_one_batch(model, X, optimizer, beta):
    criterion = nn.MSELoss()
    l1 = nn.L1Loss()

    model.train()
    optimizer.zero_grad()
    pred = model(X)

    # === Logic Loss y hệt FedLocalTrain ===
    if isinstance(model, VAELSTM):
        recon_x, mu, logvar = pred
        recon_loss = criterion(recon_x, X)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        mse_v = recon_loss.item()
        mae_v = l1(recon_x, X).item()

    elif isinstance(model, VAE):
        recon_x, mu, logvar = pred
        recon_loss = criterion(recon_x, X)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        mse_v = recon_loss.item()
        mae_v = l1(recon_x, X).item()

    elif isinstance(model, (AELSTM, AECNN)):
        recon = pred
        loss = criterion(recon, X)
        mse_v = loss.item()
        mae_v = l1(recon, X).item()

    elif isinstance(model, LSTMWithIForest):
        feats = pred
        loss = (feats ** 2).mean()
        mse_v = loss.item()
        mae_v = torch.mean(torch.abs(feats)).item()


    elif isinstance(model, LSTMModel):
        pred_shifted = pred[:, :-1]
        target_shifted = X[:, 1:]
        loss = criterion(pred_shifted, target_shifted)
        mse_v = loss.item()
        mae_v = l1(pred_shifted, target_shifted).item()

    loss.backward()
    optimizer.step()

    return loss.item(), mse_v, mae_v



def run_centralized_training(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BẮT ĐẦU CENTRALIZED TRAINING ({cfg['model']['type']}) trên thiết bị: {device}")

    central_trainer = FedLocalTrain(
        client_id="centralized",
        data_path=FULL_DATA_PATH,
        model_type=cfg['model']['type'],
        window_len=cfg['model']['window_len'],
        overlap=cfg['model']['overlap'],
        federated_type="Centralized"
    )

    X_train, X_val, X_test = central_trainer.split_data()
    print(f"Dataset: {len(X_train)} train samples, {len(X_test)} test samples.")
    #local_epochs = cfg['model']['local_epochs']

    model = central_trainer.build_model()
    model.to(device)

    round = cfg['federated']['rounds']
    lr = cfg['training']['lr']
    beta = cfg['training'].get('beta', 0.001)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ghi header CSV
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "mae", "mse"])

    # Chuẩn bị Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    batch_size = cfg['training']['batch_size']
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- TRAIN LOOP ---
    print(f"Start Training for {round} epochs...")
    for epoch in range(1, round + 1):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_mae = 0.0
        num_batches = 0
        for batch in train_loader:
            xb = batch[0].to(device)

            loss_item, mse_item, mae_item = train_one_batch(
                model, xb, optimizer, beta
            )

            epoch_loss += loss_item
            epoch_mse += mse_item
            epoch_mae += mae_item
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        avg_train_mse = epoch_mse / num_batches
        avg_train_mae = epoch_mae / num_batches


    # B. GLOBAL EVALUATION PHASE (Mini-batch Eval)
        #
        df_real = load_and_scale_data(DATA_PATH_REAL, node_id=None)
        window_len = cfg["model"]["window_len"]
        overlap_rate = cfg["model"]["overlap"]
        _, X_val_real, _ = process_data_pipeline(df_real, window_len, overlap_rate)
        X_val_real_tensor = torch.tensor(X_val_real, dtype=torch.float32).to(device)
        eval_dataset_real = TensorDataset(X_val_real_tensor)
        eval_loader_real = DataLoader(eval_dataset_real, batch_size=batch_size, shuffle=True)


        model.eval()
        eval_batch_size = cfg["training"]["batch_size"]

        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            #for batch in test_loader: (EVAL TRÊN GIẢ LẬP)
            for batch in eval_loader_real: #(EVAL TRÊN DATA REAL)
                xb = batch[0]

                loss_val, mse, mae, _ = compute_loss(model, xb, beta=beta)

                # Convert tensor → float
                if isinstance(loss_val, torch.Tensor): loss_val = loss_val.item()
                if isinstance(mse, torch.Tensor): mse = mse.item()
                if isinstance(mae, torch.Tensor): mae = mae.item()

                bs = xb.size(0)
                total_loss += loss_val * bs
                total_mse += mse * bs
                total_mae += mae * bs
                total_samples += bs

        # Average toàn bộ test set
        test_loss = total_loss / total_samples
        test_mse = total_mse / total_samples
        test_mae = total_mae / total_samples
        # C. LOGGING
        if wandb.run is not None:
            wandb.log({
                #"train_loss": avg_train_loss,
                # Key "Global/..." để W&B tự động vẽ chung biểu đồ với FL Server
                #"Global/Loss": val_loss,
                "Global/MAE": test_mae,
                "Global/MSE": test_mse,
                "round": epoch
            })

        # Ghi CSV
        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, test_loss, test_mae, test_mse])

        if epoch % 2 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{round} | Train Loss: {avg_train_loss:.4f} | Global MAE: {test_mae:.4f} | Global MSE: {test_mse:.4f}")

    print("\nTraining hoàn tất.")

    # Lưu Model Pytorch
    model_path = f"{OUTPUT_DIR}/centralized_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Đã lưu model tại: {model_path}")
    print(f"Đã lưu metrics tại: {csv_path}")

    final_mse, final_mae = get_metric_eval(model_path, DATA_PATH_REAL, cfg, n_rounds = 10 )
    print(f"Metric: MAE: {final_mae:4f} | MSE: {final_mse:.4f}")
    if wandb.run is not None:
        wandb.log({
            "Final_MAE": final_mae,
            "Final_MSE": final_mse
        })



    plot_comparison()


def plot_comparison():
    fl1_csv = f"{OUTPUT_DIR}/FedAvg_global_metrics.csv"
    fl2_csv = f"{OUTPUT_DIR}/FedBN_global_metrics.csv"
    fl3_csv = f"{OUTPUT_DIR}/FedProx_global_metrics.csv"
    cen_csv = f"{OUTPUT_DIR}/centralized_metrics.csv"

    if not os.path.exists(cen_csv):
        print(f"[ERR] Không tìm thấy {cen_csv}")
        return
    if not os.path.exists(fl1_csv):
        print(f"[ERR] Không tìm thấy {fl1_csv}")
        return
    if not os.path.exists(fl2_csv):
        print(f"[ERR] Không tìm thấy {fl2_csv}")
        return
    if not os.path.exists(fl3_csv):
        print(f"[ERR] Không tìm thấy {fl3_csv}")
        return

    df_cen = pd.read_csv(cen_csv)
    df_fl1 = pd.read_csv(fl1_csv)
    df_fl2 = pd.read_csv(fl2_csv)
    df_fl3 = pd.read_csv(fl3_csv)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # SUBPLOT 1: LOSS
    axes[0].plot(df_cen['round'], df_cen['loss'], label='Centralized Loss', linestyle='--')
    axes[0].plot(df_fl1['round'], df_fl1['loss'], label='FedAvg Global Loss', color='red')
    axes[0].plot(df_fl2['round'], df_fl2['loss'], label='FedBN Global Loss', color='blue')
    axes[0].plot(df_fl3['round'], df_fl3['loss'], label='FedProx Global Loss', color='green')

    axes[0].set_title(f"Loss Comparison: Centralized vs Global")
    axes[0].set_xlabel("Round / Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # SUBPLOT 2: MAE
    axes[1].plot(df_cen['round'], df_cen['mae'], label='Centralized MAE', linestyle='--')
    axes[1].plot(df_fl1['round'], df_fl1['mae'], label='FedAvg Global MAE', color = 'red')
    axes[1].plot(df_fl2['round'], df_fl2['mae'], label='FedBN Global MAE', color='blue')
    axes[1].plot(df_fl3['round'], df_fl3['mae'], label='FedProx Global MAE', color='green')
    axes[1].set_title(f"MSE Comparison: Centralized vs Federated")
    axes[1].set_xlabel("Round / Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # Sắp xếp layout gọn
    plt.tight_layout()

    save_path = f"{OUTPUT_DIR}/comparison_loss_mse.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Đã lưu tại: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=f"RunCentralize_{cfg['model']['type']}_real",
                        help='Tên định danh chung cho cả Server và Client')
    parser.add_argument('--note', type=str, default='', help='Ghi chú lần chạy')
    args = parser.parse_args()

    print(f" Đang khởi tạo WandB với Group: {args.run_name}")
    wandb.init(
        project="FedML_KPI_Project",
        group=args.run_name,
        job_type="centralized",
        name=f"Centralize_{args.run_name}",
        config=cfg['federated'],
        notes=args.note
    )
    run_centralized_training(cfg, args.run_name)

