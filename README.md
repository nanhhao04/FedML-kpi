# Hướng dẫn chạy Federated Learning và Benchmark

## Bước 1: Chạy Federated Learning

### 1. Khởi động Server
Mở các terminal và chạy:
```bash
python server.py

# Sửa số lượng client ở trong config.yml (ví dụ 3)
python client.py --id 1

python client.py --id 2

python client.py --id 3

```

## Bước 2. Benchmark so sánh mô hình global và local
Mở các terminal và chạy:
```bash
# Thay 1 thành 2,3,...
python benchmark.py --id 1  
```

