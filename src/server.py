import csv
import os
import threading
import grpc
import time
import torch
from concurrent import futures
import yaml
from threading import Lock
import fedfl_pb2
import fedfl_pb2_grpc
import pandas as pd
from fedml_logic import serialize_weights, aggregate_weights_avg, aggregate_weights_fedbn
from fedml_logic import FedServerLogic, deserialize_weights
from utils import get_metric_tail, get_metric_eval
import argparse
import wandb

# Load cấu hình
cfg = yaml.safe_load(open("config.yml"))



class FederationServicer(fedfl_pb2_grpc.FederationServiceServicer):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rounds = cfg['federated']['rounds']
        self.num_clients = cfg['federated']['clients']
        self.federated_type = cfg['federated']['type']  # FedAvg, FedBN, FedProx
        self.current_round = 1

        self.server_logic = FedServerLogic(cfg)
        self.global_weights = self.server_logic.global_model.state_dict()

        self.local_updates = []
        self.lock = Lock()

        self.client_status = {i: False for i in range(1, self.num_clients + 1)}

        self.total_comm_cost = 0.0
        self.fl_start_time = time.time()
        self.fl_finished = threading.Event()

    def JoinFederation(self, request, context):
        with self.lock:
            # Đánh dấu client đã join
            self.client_status[request.client_id] = True
            print(f"[SERVER] Client {request.client_id} đã kết nối.")

            # Kiểm tra số lượng client
            joined = sum(self.client_status.values())
            print(f"[SERVER] Trạng thái: {joined}/{self.num_clients} client đã sẵn sàng.")

            # Nếu chưa đủ client -> chờ
            if joined < self.num_clients:
                return fedfl_pb2.WeightsResponse(
                    serialized_weights=b"",
                    round_id=0,
                    is_final=False,
                    wait_join=True
                )

            print(f"[SERVER] Đủ client. Bắt đầu Round 1 với chiến lược: {self.federated_type}!")

            # Tính chi phí gửi model khởi tạo cho client này (Download Cost)
            init_weights_bytes = serialize_weights(self.global_weights)
            self.total_comm_cost += len(init_weights_bytes) / (1024 * 1024)

            return fedfl_pb2.WeightsResponse(
                serialized_weights=init_weights_bytes,
                round_id=self.current_round,
                is_final=False,
                wait_join=False
            )

    def SendLocalUpdate(self, request, context):
        with self.lock:

            # Xử lý trường hợp client polling khi đang chờ
            if self.current_round == 0:
                return fedfl_pb2.WeightsResponse(wait_join=True)

            if self.current_round > self.rounds:
                return fedfl_pb2.WeightsResponse(is_final=True)

            client_id = request.client_id

            # Tính toán Communication Cost
            upload_size_bytes = len(request.serialized_weights)
            upload_size_mb = upload_size_bytes / (1024 * 1024)
            self.total_comm_cost += upload_size_mb

            # In Metrics từ Client
            print(f"[Round {self.current_round}] Client {request.client_id}: "
                  f"Loss={request.train_loss:.4f} | "
                  f"MAE={request.train_mae:.4f} | "
                  f"MSE={request.train_mse:.4f} | "
                  f"Upload={upload_size_mb:.2f} MB")

            # Thu thập cập nhật
            local_weights = deserialize_weights(request.serialized_weights)
            self.local_updates.append((local_weights, request.num_samples))

            # Kiểm tra điều kiện tổng hợp (Aggregation)
            if len(self.local_updates) < self.num_clients:
                return fedfl_pb2.WeightsResponse(
                    serialized_weights=b"",  # Empty - signal to wait
                    round_id=self.current_round,
                    is_final=False
                )

            # Đủ clients - thực hiện aggregation
            print(f"\n--- [SERVER] Aggregating Round {self.current_round}/{self.rounds} ({self.federated_type}) ---")

            # Chọn thuật toán Aggregation dựa trên Config
            if self.federated_type == "FedBN":
                new_global_weights = aggregate_weights_fedbn(self.local_updates)
            else:
                new_global_weights = aggregate_weights_avg(self.local_updates)

            self.global_weights = new_global_weights

            # Evaluate trên test full
            self.server_logic.set_global_state(self.global_weights)
            try:
                g_metrics = self.server_logic.evaluate_global()

                print(f"\033[92m[GLOBAL BENCHMARK] Round {self.current_round} Result: "
                      #f"Loss: {g_metrics.get('loss', 0):.4f} | "
                      f"MAE : {g_metrics.get('mae', 0):.4f} | "
                      f"MSE : {g_metrics.get('mse', 0):.4f}\033[0m")

                os.makedirs("../outputs", exist_ok=True)
                csv_path = f"../outputs/{cfg['federated']['type']}_global_metrics.csv"

                if wandb.run is not None:
                    wandb.log({
                        "Global/MAE": g_metrics.get('mae', 0),
                        "Global/MSE": g_metrics.get('mse', 0),
                        "round": self.current_round
                    })

                # Chỉ xóa khi bắt đầu chạy FL
                if self.current_round == 1 and os.path.exists(csv_path):
                    os.remove(csv_path)

                file_exists = os.path.isfile(csv_path)

                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["round", "loss", "mae", "mse"])
                    writer.writerow([
                        self.current_round,
                        g_metrics.get('loss', 0),
                        g_metrics.get('mae', 0),
                        g_metrics.get('mse', 0)
                    ])

                print(f"[SERVER] Đã lưu metric round {self.current_round} vào {csv_path}")
            except Exception as e:
                print(f"\033[91m[WARNING] Không thể evaluate global model: {e}\033[0m")

            # Tính Communication Cost
            global_weights_bytes = serialize_weights(self.global_weights)
            broadcast_size_mb = (len(global_weights_bytes) * self.num_clients) / (1024 * 1024)
            self.total_comm_cost += broadcast_size_mb

            print(f"[SERVER] Tổng chi phí truyền thông tích lũy: {self.total_comm_cost:.2f} MB")

            # Chuyển sang vòng mới
            self.current_round += 1
            self.local_updates = []  # Reset for next round

            # Kiểm tra nếu đã hoàn thành
            if self.current_round > self.rounds:
                print("\n" + "=" * 70)
                print("[SERVER] Quá trình FL hoàn thành!")
                print("=" * 70)

                t1 = time.time()
                total_time = (t1 - self.fl_start_time)
                print(f"Tổng thời gian FL: {total_time:.2f} s")
                if wandb.run is not None:
                    wandb.log({"Time": total_time})

                # Lưu global model final
                final_model_path = "../outputs/global_model_final.pth"
                torch.save(self.global_weights, final_model_path)
                print(f" Saved final global model: {final_model_path}")
                print(f" Total communication cost: {self.total_comm_cost:.2f} MB")
                print("=" * 70 + "\n")

                # In ra metric avg 10 round cuối
                #data_path = f"../data/data_hl19_full.csv"
                #model_path = f"../outputs/global_model_final.pth"
                #final_mse, final_mae = get_metric_eval(model_path, data_path, cfg, n_rounds=10)
                #print(f"Final Eval Metric: MAE: {final_mae:.4f} | MSE: {final_mse:.4f}")
                #if wandb.run is not None:
                #   wandb.log({"mae": final_mae, "mse": final_mse})
                self.fl_finished.set()
                return fedfl_pb2.WeightsResponse(is_final=True)

            # Trả về global weights cho round mới
            return fedfl_pb2.WeightsResponse(
                serialized_weights=serialize_weights(self.global_weights),
                round_id=self.current_round,
                is_final=False
            )



def serve():
    import os
    os.makedirs("../outputs", exist_ok=True)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Tạo servicer riêng để truy cập biến fl_finished
    servicer = FederationServicer(cfg)
    fedfl_pb2_grpc.add_FederationServiceServicer_to_server(servicer, server)

    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    print(f"Server gRPC đang chạy trên cổng 50051...")

    try:
        # Chờ cờ báo hiệu kết thúc từ Servicer
        while not servicer.fl_finished.is_set():
            time.sleep(1)

        print("\n[MAIN THREAD] Phát hiện quá trình FL đã kết thúc.")
        print("waiting...")
        time.sleep(3)

        server.stop(0)
        print("Server gRPC đã đóng.")

        # --- FINAL EVALUATION ---
        print("\n" + "=" * 30)
        print("TỔNG HỢP KẾT QUẢ CUỐI CÙNG")
        print("=" * 30)

        # Logic rẽ nhánh cho FedBN
        '''
        if cfg['federated']['type'] == "FedBN":
            print("[INFO] FedBN Detected: Sử dụng metric trung bình 10 round cuối (Tail Metric).")
            # Đường dẫn file CSV log
            metric_csv_path = f"../outputs/{cfg['federated']['type']}_global_metrics.csv"

            if os.path.exists(metric_csv_path):
                # Lưu ý: get_metric_tail trả về (mae, mse)
                mae, mse = get_metric_tail(metric_csv_path, n_tail=10)
                print(f"AVG Tail (Last 10 Rounds) -> MAE: {mae:.4f} | MSE: {mse:.4f}")
            else:
                print(f"[WARN] Không tìm thấy file metrics tại {metric_csv_path}")
                mae, mse = 0.0, 0.0

        else:
        '''
        print("Running Full Evaluation on Test Set...")
        data_path = "../data/data_hl19_full.csv"
        model_path = "../outputs/global_model_final.pth"

        mse, mae = get_metric_eval(model_path, data_path, cfg, n_rounds=10)
        print(f"Full Eval Result : MAE: {mae:.4f} | MSE: {mse:.4f}")

        # Log WandB kết quả cuối cùng
        if wandb.run is not None:
            wandb.log({"Final_MAE": mae, "Final_MSE": mse})
            wandb.finish()

    except KeyboardInterrupt:
        server.stop(0)
        print("\nServer đã dừng bởi người dùng.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=f"{cfg['federated']['type']}_{cfg['model']['type']}",
                        help='Tên định danh chung cho cả Server và Client')
    parser.add_argument('--note', type=str, default='', help='Ghi chú lần chạy')
    args = parser.parse_args()

    print(f" Đang khởi tạo WandB với Group: {args.run_name}")
    wandb.init(
        project="FedML_KPI_Project",
        group=args.run_name,
        job_type="server",
        name=f"Server_{args.run_name}",
        config=cfg['federated'],
        notes=args.note
    )

    serve()

