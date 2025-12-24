# Hướng dẫn chạy Federated Learning và Benchmark
So sánh hiệu năng của các phương pháp federated khác nhau (FedAvg, FedProx, FedBN) trên các mô hình dạng lstm và autoencoder cho detect anomaly tập dữ liệu kpi mạng.

## Bước 1: Push data vào folder data

## Bước 2: Chạy Federated Learning

### 1. Khởi động Server
Mở các terminal và chạy:
```bash
python server.py

# Sửa số lượng client ở trong config.yml (ví dụ 3)
python client.py --id 1

python client.py --id 2

python client.py --id 3

```
## Bước 3. Chạy full local
```bash
python ./run_centralized.py
```


