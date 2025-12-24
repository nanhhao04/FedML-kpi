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
import argparse
import time
import wandb
cfg = yaml.safe_load(open("config.yml", "r"))
os.makedirs("../outputs", exist_ok=True)


def run_client(client_id, server_address="localhost:50051"):
    data_path = f"../data/data_hl19_node_{client_id}.csv"
    fed_type = cfg['federated']['type']
    client_local = FedLocalTrain(
        client_id=client_id,
        data_path=data_path,
        model_type=cfg['model']['type'],
        window_len=cfg['model']['window_len'],
        overlap=cfg['model']['overlap'],
        federated_type=cfg['federated']['type']
    )

    if client_local._get_input_dim() == 0:
        print(f"Client {client_id}: Không có dữ liệu.")
        return

    X_train, X_val, X_test = client_local.split_data()
    num_samples = len(X_train)

    # Model cục bộ
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

        # Nhận Global Weights ban đầu
        if resp.serialized_weights:
            global_weights = deserialize_weights(resp.serialized_weights)

            if fed_type == "FedBN":
                current_state = model.state_dict()
                bn_identifiers = ["bn", "norm"]

                for key, value in global_weights.items():
                    # Chỉ load những key ko chứa từ khóa 'bn' hay 'norm'
                    if not any(idv in key for idv in bn_identifiers):
                        current_state[key] = value

                # current_state đã giữ nguyên BN cũ
                model.load_state_dict(current_state, strict=False)
            else:
                model.load_state_dict(global_weights)


        current_round = resp.round_id
        is_final = resp.is_final

        # Training Loop
        while not is_final:
            print(f"[Client {client_id}] Round {current_round} ({fed_type})...")

            # Cập nhật global weights vào model local trước khi train
            if fed_type == "FedBN":
                current_state = model.state_dict()
                for key, value in global_weights.items():
                    if not any(bn_key in key for bn_key in [
                        '.running_mean',
                        '.running_var',
                        '.num_batches_tracked'
                    ]):
                        current_state[key] = value
                model.load_state_dict(current_state, strict=False)
            else:
                # FedAvg / FedProx: Update toàn bộ
                model.load_state_dict(global_weights)

            global_weights_final = model.state_dict()  # Lưu backup

            try:
                c_round_int = int(current_round)
            except:
                c_round_int = 0

            # Train cục bộ
            # Lưu ý: global_weights được truyền vào get_local_update để dùng tính Loss cho FedProx
            local_weights, num_samples, metrics = client_local.get_local_update(
                global_weights=global_weights,
                current_round=c_round_int
            )

            # Cập nhật model local với weights mới train xong (để chuẩn bị gửi)
            if local_weights is not None:
                model.load_state_dict(local_weights)

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

            print(
                f"  -> Loss={metrics['train_loss']:.4f}, MAE={metrics['train_mae']:.4f}, MSE={metrics['train_mse']:.4f}")
            print("Đang gửi cập nhật lên Server...")

            # Serialize weights để gửi , FedBN: Local update vẫn gửi full weights lên, Server sẽ lọc sau.
            weights_to_send = serialize_weights(local_weights)

            update_req = fedfl_pb2.LocalUpdateRequest(
                client_id=client_id,
                serialized_weights=weights_to_send,
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
                # Nhận Global Weights mới từ Server (chưa load vào model ngay, vòng lặp sau sẽ load)
                global_weights = deserialize_weights(resp.serialized_weights)

            current_round = resp.round_id
            is_final = resp.is_final
            time.sleep(0.3)

        print(f"Client {client_id}: FL hoàn thành. Bắt đầu đánh giá...")

        #torch.save(global_weights_final, f"outputs/global_weights_client_{client_id}.pth")
        #print(f"Saved global weights: outputs/global_weights_client_{client_id}.pth")

        # Load weights cuối cùng để evaluate
        if fed_type == "FedBN":
            current_state = model.state_dict()
            for key, value in global_weights.items():
                if not any(bn_key in key for bn_key in [
                    '.running_mean', '.running_var', '.num_batches_tracked'
                ]):
                    current_state[key] = value
            model.load_state_dict(current_state, strict=False)
        else:
            model.load_state_dict(global_weights)

        model.eval()

        # Anomaly Score bằng compute_loss()
        X_pred = client_local.X_all
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

        # Export CSV
        N = 300
        all_data = client_local.dates_df.iloc[-len(scores):].copy()
        all_data["anomaly_score"] = scores

        top_anomalies = all_data.nlargest(N, "anomaly_score")
        top_anomalies.to_csv(f"../outputs/fl_node_{client_id}_final_scores.csv", index=False)

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
        plt.title(f"FL Anomaly Score - Node {client_id} ({fed_type})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"../outputs/fl_node_{client_id}_anomaly_plot.png", dpi=100)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Client ID (1, 2, 3...)")
    parser.add_argument("--server", type=str, default="localhost:50051", help="Server Address")
    parser.add_argument('--run_name', type=str, default=f"{cfg['federated']['type']}_{cfg['model']['type']}",
                        help='Tên định danh chung cho cả Server và Client')
    parser.add_argument('--note', type=str, default='', help='Ghi chú lần chạy')
    args = parser.parse_args()

    MAX_RETRIES = 50
    RETRY_DELAY = 5

    print(f"--- Client {args.id} đang khởi động ---")
    print(f"Mục tiêu: {args.server}")

    for attempt in range(MAX_RETRIES):
        try:
            print(f" Đang khởi tạo WandB với Group: {args.run_name}")
            '''
            wandb.init(
                project="FedML_KPI_Project",
                group=args.run_name,
                job_type="client",
                name=f"Client_{args.run_name}",
                config=cfg['federated'],
                notes=args.note
            )
            '''
            run_client(client_id=args.id, server_address=args.server)
            break


        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print(
                    f"[CẢNH BÁO] Không kết nối được Server. Đang thử lại sau {RETRY_DELAY}s... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[LỖI] Xảy ra lỗi gRPC khác: {e}")
                break

        except Exception as ex:
            print(f"[INFO] Server chưa sẵn sàng ({ex}). Đợi {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    else:
        print("[THẤT BẠI] Đã thử quá số lần quy định. Vui lòng kiểm tra Server.")


