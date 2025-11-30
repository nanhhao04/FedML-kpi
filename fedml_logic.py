# fedml_logic.py
import torch
import torch.nn as nn
import torch.optim as optim
import io
import yaml
import numpy as np
# Import Scheduler
from torch.optim.lr_scheduler import StepLR
from preprocess import load_and_scale_data, create_sequence_with_date, split_train_val_test
from models import LSTMModel, AELSTM, LSTMWithIForest, VAELSTM

cfg = yaml.safe_load(open("config.yml"))


def serialize_weights(weights_dict):
    buffer = io.BytesIO()
    torch.save(weights_dict, buffer)
    return buffer.getvalue()


def deserialize_weights(weights_bytes):
    if not weights_bytes:
        return None

    buffer = io.BytesIO(weights_bytes)

    try:
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"[Deserialize] Lỗi: {e}")


def aggregate_weights(local_updates):
    if not local_updates:
        return None

    sizes = [n for _, n in local_updates]
    total = sum(sizes)

    first = local_updates[0][0]
    new_global = {}

    for key in first.keys():
        new_global[key] = torch.zeros_like(first[key])
        for w, n in local_updates:
            new_global[key] += w[key] * (n / total)

    return new_global


class FedLocalTrain:
    def __init__(self, client_id, model_type, data_path, window_len, overlap):
        self.id = client_id
        self.model_type = model_type
        self.data_path = data_path
        self.window_len = window_len
        self.overlap = overlap

        self.df = load_and_scale_data(self.data_path, self.id)
        self.X, self.dates_df = create_sequence_with_date(
            self.df, window_len=self.window_len, overlap_rate=self.overlap
        )

        self._input_dim = 0 if len(self.X) == 0 else self.X.shape[-1]

        # --- [NEW] Lưu trạng thái Optimizer cho Local Benchmark ---
        self.local_optimizer = None
        self.local_scheduler = None

    def _get_input_dim(self):
        return self._input_dim

    def build_model(self):
        inp = self._get_input_dim()
        if inp == 0:
            return None

        if self.model_type == "lstm":
            return LSTMModel(input_dim=inp)
        if self.model_type == "ae_lstm":
            return AELSTM(input_dim=inp)
        if self.model_type == "iforest_lstm":
            return LSTMWithIForest(input_dim=inp)
        if self.model_type == "vae_lstm":
            return VAELSTM(input_dim=inp)

    def split_data(self):
        return split_train_val_test(self.X, train_ratio=0.7, val_ratio=0.15)

    # Train Model cho FL (Có thêm Decay LR theo Round)
    def train_model(self, model, X_train, current_round=0):
        epochs = 1
        base_lr = cfg["training"]["lr"]
        beta = cfg["training"].get("beta", 0.001)

        # Sau mỗi 10 round, LR giảm còn 90% (nhân 0.9)
        decay_rate = 0.9
        decay_step = 10

        if current_round > 0:
            current_lr = base_lr * (decay_rate ** (current_round // decay_step))
        else:
            current_lr = base_lr

        criterion = nn.MSELoss()
        l1 = nn.L1Loss()

        optimzr = optim.Adam(model.parameters(), lr=current_lr)
        model.train()

        X = torch.tensor(X_train, dtype=torch.float32)
        input_dim = X.shape[-1]

        last_loss = 0.0
        last_mae = 0.0
        last_mse = 0.0

        for _ in range(epochs):
            optimzr.zero_grad()
            pred = model(X)

            # Tính metric
            if isinstance(model, VAELSTM):
                recon_x, mu, logvar = pred
                recon_loss = criterion(recon_x, X)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss
                mse_v = criterion(recon_x, X).item()
                mae_v = l1(recon_x, X).item()

            elif isinstance(model, AELSTM):
                loss = criterion(pred, X)
                mse_v = loss.item()
                mae_v = l1(pred, X).item()

            elif isinstance(model, LSTMWithIForest):
                feats = pred
                loss = (feats ** 2).mean()
                mse_v = criterion(feats, torch.zeros_like(feats)).item()
                mae_v = l1(feats, torch.zeros_like(feats)).item()

            else:
                if pred.ndim == 3: pred = pred[:, -1, :]
                if pred.shape[-1] == input_dim:
                    target = X[:, -1, :]
                else:
                    target = X[:, -1, 0].unsqueeze(1)

                loss = criterion(pred, target)
                mse_v = loss.item()
                mae_v = l1(pred, target).item()

            loss.backward()
            optimzr.step()

            last_loss = loss.item()
            last_mse = mse_v
            last_mae = mae_v

        metrics = {
            "train_loss": last_loss,
            "train_mae": last_mae,
            "train_mse": last_mse
        }

        return model, metrics

    def get_local_update(self, global_weights=None, current_round=0):
        if self._get_input_dim() == 0:
            return None, 0, {"train_loss": 0.0, "train_mae": 0.0, "train_mse": 0.0}

        X_train, _, _ = self.split_data()
        local_model = self.build_model()

        if local_model is None:
            return None, 0, {"train_loss": 0.0, "train_mae": 0.0, "train_mse": 0.0}

        # Load global weights
        if global_weights is not None:
            local_model.load_state_dict(global_weights)

        # Truyền current_round vào để chỉnh LR
        trained_model, metrics = self.train_model(local_model, X_train, current_round=current_round)

        return trained_model.state_dict(), len(X_train), metrics

    # Train 1 epoch cho LOCAL BENCHMARK
    def compute_local_train_step(self, model, X_train):

        criterion = nn.MSELoss()
        l1 = nn.L1Loss()
        lr = cfg["training"]["lr"]
        beta = cfg["training"].get("beta", 0.001)

        if self.local_optimizer is None:
            self.local_optimizer = optim.Adam(model.parameters(), lr=lr)
            # Thêm Scheduler: Giảm LR đi sau mỗi 10 epoch
            self.local_scheduler = StepLR(self.local_optimizer, step_size=10, gamma=0.1)


        model.train()
        X = torch.tensor(X_train, dtype=torch.float32)
        input_dim = X.shape[-1]

        self.local_optimizer.zero_grad()
        pred = model(X)

        if isinstance(model, VAELSTM):
            recon_x, mu, logvar = pred
            recon_loss = criterion(recon_x, X)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
            mse_v = criterion(recon_x, X).item()
            mae_v = l1(recon_x, X).item()

        elif isinstance(model, AELSTM):
            loss = criterion(pred, X)
            mse_v = loss.item()
            mae_v = l1(pred, X).item()

        elif isinstance(model, LSTMWithIForest):
            feats = pred
            loss = (feats ** 2).mean()
            mse_v = criterion(feats, torch.zeros_like(feats)).item()
            mae_v = l1(feats, torch.zeros_like(feats)).item()

        else:
            if pred.ndim == 3: pred = pred[:, -1, :]
            if pred.shape[-1] == input_dim:
                target = X[:, -1, :]
            else:
                target = X[:, -1, 0].unsqueeze(1)
            loss = criterion(pred, target)
            mse_v = loss.item()
            mae_v = l1(pred, target).item()

        loss.backward()
        self.local_optimizer.step()

        self.local_scheduler.step()

        return model, {
            "train_loss": loss.item(),
            "train_mse": mse_v,
            "train_mae": mae_v
        }

    #  Evaluate
    def compute_metrics(self, model, X_test, X_train=None):
        criterion = nn.MSELoss()
        l1 = nn.L1Loss()
        model.eval()
        X = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            pred = model(X)

        input_dim = X.shape[-1]

        if isinstance(model, VAELSTM):
            recon_x, mu, logvar = pred
            mse = criterion(recon_x, X).item()
            mae = l1(recon_x, X).item()
            loss = mse
            scores = ((recon_x - X) ** 2).mean(dim=(1, 2)).numpy()

        elif isinstance(model, AELSTM):
            mse = criterion(pred, X).item()
            mae = l1(pred, X).item()
            loss = mse
            scores = ((pred - X) ** 2).mean(dim=(1, 2)).numpy()

        elif isinstance(model, LSTMWithIForest):
            feats_tr = model(torch.tensor(X_train, dtype=torch.float32))
            model.fit_iforest(feats_tr)
            feats_test = model(X)
            scores = model.anomaly_score(feats_test)
            mse = np.mean(scores)
            mae = mse
            loss = mse

        else:
            if pred.ndim == 3: pred = pred[:, -1, :]
            if pred.shape[-1] == input_dim:
                tgt = X[:, -1, :]
            else:
                tgt = X[:, -1, 0].unsqueeze(1)
            mse = criterion(pred, tgt).item()
            mae = l1(pred, tgt).item()
            loss = mse
            scores = ((pred - tgt) ** 2).mean(dim=1).numpy()

        return {
            "mse": mse,
            "mae": mae,
            "loss": loss,
            "scores": scores
        }


class FedServerLogic:
    def __init__(self, cfg):
        self.model_type = cfg["model"]["type"]
        self.window_len = cfg["model"]["window_len"]
        self.overlap = cfg["model"]["overlap"]

        temp = FedLocalTrain(
            client_id=1,
            data_path=f"data/data_hl19_node_1.csv",
            model_type=self.model_type,
            window_len=self.window_len,
            overlap=self.overlap
        )

        inp = temp._get_input_dim()
        if inp == 0:
            raise ValueError("Client 1 không có dữ liệu!")

        self.global_model = self._build_global_model(inp)
        self.input_dim = inp

        print(f"[ServerLogic] Built global model (input_dim={inp})")

    def _build_global_model(self, inp):
        if self.model_type == "lstm":
            return LSTMModel(input_dim=inp)
        if self.model_type == "ae_lstm":
            return AELSTM(input_dim=inp)
        if self.model_type == "iforest_lstm":
            return LSTMWithIForest(input_dim=inp)
        if self.model_type == "vae_lstm":
            return VAELSTM(input_dim=inp)