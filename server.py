import grpc
import time
import torch
from concurrent import futures
import yaml
from threading import Lock
import fedfl_pb2
import fedfl_pb2_grpc
from fedml_logic import serialize_weights, aggregate_weights
from fedml_logic import FedServerLogic, deserialize_weights

# Load cấu hình
cfg = yaml.safe_load(open("config.yml"))


class FederationServicer(fedfl_pb2_grpc.FederationServiceServicer):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rounds = cfg['federated']['rounds']
        self.num_clients = cfg['federated']['clients']
        self.current_round = 1

        self.server_logic = FedServerLogic(cfg)
        self.global_weights = self.server_logic.global_model.state_dict()

        self.local_updates = []
        self.lock = Lock()

        self.client_status = {i: False for i in range(1, self.num_clients + 1)}

        self.total_comm_cost = 0.0

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

            print("[SERVER] Đủ client, bắt đầu Round 1!")

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
            if len(self.local_updates) == self.num_clients:
                print(f"\n--- [SERVER] Aggregating Round {self.current_round}/{self.rounds} ---")

                # FedAvg
                new_global_weights = aggregate_weights(self.local_updates)
                self.global_weights = new_global_weights

                # Tính Communication Cost
                global_weights_bytes = serialize_weights(self.global_weights)
                broadcast_size_mb = (len(global_weights_bytes) * self.num_clients) / (1024 * 1024)
                self.total_comm_cost += broadcast_size_mb

                print(f"[SERVER] Tổng chi phí truyền thông tích lũy: {self.total_comm_cost:.2f} MB")

                # Chuyển sang vòng mới
                self.current_round += 1
                self.local_updates = []

                # lưu global model
                if self.current_round > self.rounds:
                    print("\n" + "=" * 70)
                    print("[SERVER] Quá trình FL hoàn thành!")
                    print("=" * 70)

                    # Lưu global model final
                    final_model_path = "outputs/global_model_final.pth"
                    torch.save(self.global_weights, final_model_path)
                    print(f" Saved final global model: {final_model_path}")
                    print(f" Total communication cost: {self.total_comm_cost:.2f} MB")
                    print("=" * 70 + "\n")

                    return fedfl_pb2.WeightsResponse(is_final=True)

            # Trả về phản hồi
            return fedfl_pb2.WeightsResponse(
                serialized_weights=serialize_weights(self.global_weights),
                round_id=self.current_round,
                is_final=False
            )


def serve():
    # Tạo thư mục outputs nếu chưa có
    import os
    os.makedirs("outputs", exist_ok=True)

    # Tăng max_workers nếu số lượng client lớn
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fedfl_pb2_grpc.add_FederationServiceServicer_to_server(
        FederationServicer(cfg), server
    )

    # Mở port 50051
    server.add_insecure_port('[::]:50051')
    server.start()
    print(f"Server gRPC đang chạy trên cổng 50051...")
    print(f"Cấu hình: {cfg['federated']['clients']} Clients, {cfg['federated']['rounds']} Rounds")

    try:
        while True:
            time.sleep(86400)  # Giữ server sống
    except KeyboardInterrupt:
        server.stop(0)
        print("\nServer gRPC đã dừng.")


if __name__ == '__main__':
    serve()