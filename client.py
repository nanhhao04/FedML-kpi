import argparse
import time
import grpc
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import fedfl_pb2
import fedfl_pb2_grpc
from fedml_logic import (
    FedLocalTrain,
    serialize_weights,
    deserialize_weights,
    compute_loss
)

cfg = yaml.safe_load(open("config.yml", "r"))
os.makedirs("outputs", exist_ok=True)


def run_client(client_id, server_address="localhost:50051"):
    data_path = f"data/data_hl19_node_{client_id}.csv"

    client_local = FedLocalTrain(
        client_id=client_id,
        data_path=data_path,
        model_type=cfg['model']['type'],
        window_len=cfg['model']['window_len'],
        overlap=cfg['model']['overlap']
    )

    if client_local._get_input_dim() == 0:
        print(f"Client {client_id}: Không có dữ liệu.")
        return

    X_train, X_val, X_test = client_local.split_data()
    num_samples = len(X_train)

    model = client_local.build_model()
    global_weights_final = None

    # Log metrics
    round_logs = []

    with grpc.insecure_channel(server_address) as channel:
        stub = fedfl_pb2_grpc.FederationServiceStub(channel)

        join_req = fedfl_pb2.JoinRequest(client_id=client_id, num_samples=num_samples)
        resp = stub.JoinFederation(join_req)

        while getattr(resp, "wait_join", False):
            print(f"Client {client_id}: Server chưa đủ client, đang chờ...")
            time.sleep(2)
            resp = stub.JoinFederation(join_req)

        global_weights = deserialize_weights(resp.serialized_weights) if resp.serialized_weights else model.state_dict()

        current_round = resp.round_id
        is_final = resp.is_final

        # ===== Training Loop =====
        while not is_final:
            print(f"[Client {client_id}] Round {current_round}...")
            global_weights_final = global_weights

            try:
                c_round_int = int(current_round)
            except:
                c_round_int = 0

            local_weights, num_samples, metrics = client_local.get_local_update(global_weights, current_round=c_round_int)

            try:
                round_num = int(current_round)
            except:
                round_num = None

            if round_num is not None and not any(log['round'] == round_num for log in round_logs):
                round_logs.append({
                    "round": round_num,
                    "loss": metrics.get("train_loss", 0.0),
                    "mae": metrics.get("train_mae", 0.0),
                    "mse": metrics.get("train_mse", 0.0)
                })

            print(f"  -> Loss={metrics['train_loss']:.4f}, MAE={metrics['train_mae']:.4f}, MSE={metrics['train_mse']:.4f}")
            print("Đang gửi cập nhật lên Server...")

            update_req = fedfl_pb2.LocalUpdateRequest(
                client_id=client_id,
                serialized_weights=serialize_weights(local_weights),
                num_samples=num_samples,
                train_loss=metrics['train_loss'],
                train_mae=metrics['train_mae'],
                train_mse=metrics['train_mse']
            )

            try:
                resp = stub.SendLocalUpdate(update_req)
            except Exception as e:
                print("[Lỗi kết nối] Server có thể đã tắt:", e)
                return

            if resp.serialized_weights:
                global_weights = deserialize_weights(resp.serialized_weights)

            current_round = resp.round_id
            is_final = resp.is_final
            time.sleep(0.3)

        print(f"Client {client_id}: FL hoàn thành. Bắt đầu đánh giá...")

        torch.save(global_weights_final, f"outputs/global_weights_client_{client_id}.pth")
        print(f"Saved global weights: outputs/global_weights_client_{client_id}.pth")

        model.load_state_dict(global_weights_final)
        model.eval()

        # ===== Anomaly Score bằng compute_loss() (logic thống nhất) =====
        X_pred = client_local.X
        X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32)

        if client_local.model_type == "iforest_lstm":
            # Fit IForest on train features
            feats_train = model(torch.tensor(X_train, dtype=torch.float32))
            model.fit_iforest(feats_train)
            feats_pred = model(X_pred_tensor)
            scores = model.anomaly_score(feats_pred)
        else:
            _, _, _, scores_tensor = compute_loss(model, X_pred_tensor)
            scores = scores_tensor.cpu().numpy()

        # ===== Export CSV =====
        N = 300
        all_data = client_local.dates_df.iloc[-len(scores):].copy()
        all_data["anomaly_score"] = scores

        top_anomalies = all_data.nlargest(N, "anomaly_score")
        top_anomalies.to_csv(f"outputs/fl_node_{client_id}_final_scores.csv", index=False)

        # ===== Plot Anomaly Curve =====
        plt.figure(figsize=(14, 6))
        if "date" in all_data.columns:
            all_data["date"] = pd.to_datetime(all_data["date"])
            all_data = all_data.sort_values("date")
            x_vals = all_data["date"]
        else:
            x_vals = np.arange(len(all_data))

        plt.plot(x_vals, all_data["anomaly_score"], color='blue', linewidth=1, alpha=0.7, label="All Points")

        if "date" in all_data.columns:
            top_x = pd.to_datetime(top_anomalies["date"])
        else:
            top_x = top_anomalies.index

        plt.scatter(top_x, top_anomalies["anomaly_score"], color='red', s=30, alpha=0.8,
                    label=f"Top {N} Anomalies", zorder=5)

        if "date" in all_data.columns:
            plt.gcf().autofmt_xdate()
        plt.xlabel("Date")
        plt.ylabel("Anomaly Score")
        plt.title(f"FL Anomaly Score - Node {client_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"outputs/fl_node_{client_id}_anomaly_plot.png", dpi=100)
        plt.close()

        # ===== Plot Metrics =====
        total_rounds = int(cfg.get("federated", {}).get("rounds", len(round_logs) or 1))

        metrics_map = {entry['round']: entry for entry in round_logs if entry.get('round') is not None}
        rounds_x = list(range(1, total_rounds + 1))
        loss_list, mae_list, mse_list = [], [], []

        for rr in rounds_x:
            entry = metrics_map.get(rr)
            if entry is None:
                loss_list.append(np.nan)
                mae_list.append(np.nan)
                mse_list.append(np.nan)
            else:
                loss_list.append(entry['loss'])
                mae_list.append(entry['mae'])
                mse_list.append(entry['mse'])

        plt.figure(figsize=(10, 5))
        plt.plot(rounds_x, mae_list, linewidth=2, label="MAE", color='orange')
        plt.plot(rounds_x, mse_list, linewidth=2, label="MSE", color='green', linestyle='--')
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.title(f"Training Metrics - Client {client_id} (R={total_rounds})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"outputs/client_{client_id}_metrics_plot.png")
        plt.close()

        import json
        json.dump(round_logs, open(f"outputs/client_{client_id}_metrics.json", "w"), indent=2)

        print(f"[Client {client_id}] DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--server", type=str, default="localhost:50051")
    a = parser.parse_args()

    run_client(a.id, a.server)
