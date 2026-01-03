import io
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

from preprocess import load_and_scale_data, create_sequence_with_date, split_train_val_test, process_data_pipeline, \
    DataScaler
from models import LSTMModel, AELSTM, LSTMWithIForest, VAELSTM, VAE, AECNN

# Load config
cfg = yaml.safe_load(open("config.yml"))


def serialize_weights(weights_dict):
    """Serialize state_dict to bytes."""
    buffer = io.BytesIO()
    torch.save(weights_dict, buffer)
    return buffer.getvalue()


def deserialize_weights(weights_bytes):
    """Deserialize bytes to a state_dict. Return None if input empty."""
    if not weights_bytes:
        return None

    buffer = io.BytesIO(weights_bytes)
    try:
        # Try modern API first
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions don't accept weights_only
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"[Deserialize] Lỗi: {e}")


def aggregate_weights_fedbn(local_updates):
    if not local_updates:
        return None

    sizes = [n for _, n in local_updates]
    total = float(sum(sizes))
    base_w = local_updates[0][0]
    new_global = {}

    for key in base_w.keys():
        new_global[key] = torch.zeros_like(base_w[key], dtype=torch.float32)

        for w, n in local_updates:
            new_global[key] += w[key].float() * (n / total)

        if base_w[key].dtype != torch.float32:
            new_global[key] = new_global[key].to(base_w[key].dtype)

    return new_global


def aggregate_weights_avg(local_updates):
    if not local_updates:
        return None

    sizes = [n for _, n in local_updates]
    total = float(sum(sizes))
    first = local_updates[0][0]

    new_global = {}

    #Khởi tạo buffer tích lũy
    for key in first.keys():
        if first[key].dtype == torch.long:
            new_global[key] = torch.zeros_like(first[key], dtype=torch.float32)
        else:
            new_global[key] = torch.zeros_like(first[key])

    for w, n in local_updates:
        for key in new_global.keys():
            new_global[key] += w[key] * (float(n) / total)

    # Ép kiểu lại về gốc nếu cần
    for key in first.keys():
        if first[key].dtype == torch.long:
            new_global[key] = new_global[key].long()

    return new_global
'''
# FedBN - aggreate
def aggregate_weights_fedbn(local_updates):
    if not local_updates:
        return None

    # Lấy sample size từ tất cả client
    sizes = [n for _, n in local_updates]
    total = float(sum(sizes))

    # Lấy weight mẫu đầu tiên
    base_w = local_updates[0][0]

    new_global = {}

    for key in base_w.keys():
        if any(bn_key in key for bn_key in [
            '.running_mean',
            '.running_var',
            '.num_batches_tracked'
        ]):
            continue

        new_global[key] = torch.zeros_like(base_w[key], dtype=torch.float32)
        for w, n in local_updates:
            new_global[key] += w[key] * (n / total)

    return new_global



def aggregate_weights_fedbn(local_updates):
    if not local_updates:
        return None

    sizes = [n for _, n in local_updates]
    total = float(sum(sizes))
    base_w = local_updates[0][0]
    new_global = {}

    bn_identifiers = ["bn", "norm"]

    count_skipped = 0
    for key in base_w.keys():
        # Bỏ qua toàn bộ tham số của lớp BN
        if any(idv in key for idv in bn_identifiers):
            print(f"[FedBN] Skipping local param: {key}")
            count_skipped += 1
            continue

        # Chỉ cộng gộp các lớp LSTM và Linear (FC)
        new_global[key] = torch.zeros_like(base_w[key], dtype=torch.float32)
        for w, n in local_updates:
            new_global[key] += w[key].float() * (n / total)

        if base_w[key].dtype != torch.float32:
            new_global[key] = new_global[key].to(base_w[key].dtype)

    print(f"[FedBN] Aggregated {len(new_global)} parameters (Skipped {count_skipped} BN params)")
    return new_global
'''

def compute_loss(model, X, beta=None):
    if beta is None:
        beta = cfg["training"].get("beta", 0.001)

    criterion = nn.MSELoss()
    l1 = nn.L1Loss()

    model.eval()
    with torch.no_grad():
        pred = model(X)

    input_dim = X.shape[-1]

    # VAE-LSTM
    if isinstance(model, VAELSTM):
        recon_x, mu, logvar = pred
        recon_loss = criterion(recon_x, X)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        mse = recon_loss.item()
        mae = l1(recon_x, X).item()
        scores = ((recon_x - X) ** 2).mean(dim=(1, 2))
        return loss, mse, mae, scores

    # AE-LSTM
    if isinstance(model, AELSTM):
        recon = pred
        loss = criterion(recon, X)
        mse = loss.item()
        mae = l1(recon, X).item()
        scores = ((recon - X) ** 2).mean(dim=(1, 2))
        return loss, mse, mae, scores

    # LSTM iso
    if isinstance(model, LSTMWithIForest):
        feats = pred
        loss = (feats ** 2).mean()
        mse = loss.item()
        mae = torch.mean(torch.abs(feats)).item()
        scores = feats.norm(dim=1)
        return loss, mse, mae, scores

    # AECNN
    if isinstance(model, AECNN):
        recon = pred
        loss = criterion(recon, X)
        mse = loss.item()
        mae = l1(recon, X).item()
        scores = ((recon - X) ** 2).mean(dim=(1, 2))
        return loss, mse, mae, scores

    # VAE
    if isinstance(model, VAE):
        recon_x, mu, logvar = pred
        recon_loss = criterion(recon_x, X)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        mse = recon_loss.item()
        mae = l1(recon_x, X).item()
        if recon_x.ndim == 3:
            scores = ((recon_x - X) ** 2).mean(dim=(1, 2))
        else:
            scores = ((recon_x - X) ** 2).mean(dim=1)
        return loss, mse, mae, scores

    if isinstance(model, LSTMModel):
        pred_shifted = pred[:, :-1, :]
        target_shifted = X[:, 1:, :]
        loss = criterion(pred_shifted, target_shifted)
        mse = loss.item()
        mae = l1(pred_shifted, target_shifted).item()
        scores = ((pred_shifted - target_shifted) ** 2).mean(dim=(1, 2))

        return loss, mse, mae, scores



def fedprox_loss(local_model, global_weights, mu=0.001, device='cpu'):
    proximal_term = 0.0
    for name, local_param in local_model.named_parameters():
        if name in global_weights:
            global_param = global_weights[name].to(device)
            proximal_term += torch.mean((local_param - global_param.detach()) ** 2)

    return (mu / 2.0) * proximal_term



class FedLocalTrain:
    def __init__(self, client_id, model_type, data_path, window_len, federated_type, overlap):
        self.id = client_id
        self.model_type = model_type
        self.data_path = data_path
        self.window_len = window_len
        self.overlap = overlap
        self.federated_type = federated_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df = load_and_scale_data(self.data_path, self.id)

        if len(self.df) == 0:
            self._input_dim = 0
        else:
            numeric_cols = self.df.select_dtypes(include='number').columns
            self._input_dim = len(numeric_cols)

        self.local_optimizer = None

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
        if self.model_type == "vae":
            return VAE(input_dim=inp * self.window_len)
        if self.model_type == "ae_cnn":
            return AECNN(input_dim=inp, seq_len=self.window_len)
        raise ValueError(f"Unknown model_type: {self.model_type}")

        return model.to(self.device)

    def split_data(self):
        # Process pipeline
        X_train, X_val, X_test = process_data_pipeline(
            self.df, self.window_len, self.overlap
        )

        scaler = DataScaler()
        scaler.fit(self.df.iloc[:int(0.7 * len(self.df))])  # Fit on train portion
        scaled_df = scaler.transform(self.df)
        self.X_all, self.dates_df = create_sequence_with_date(
            scaled_df, self.window_len, self.overlap
        )

        return X_train, X_val, X_test

    def train_model(self, model, X_train, global_weights=None, current_round=0):
        if X_train is None or len(X_train) == 0:
            return model, {"train_loss": 0.0, "train_mae": 0.0, "train_mse": 0.0}

        local_epochs = cfg["training"].get("local_epochs", 1)

        lr = cfg["training"]["lr"]
        beta = cfg["training"].get("beta", 0.001)
        mu = cfg["training"].get("mu", 0.001)  # FedProx

        batch_size = cfg["training"]["batch_size"]

        criterion = nn.MSELoss()
        l1 = nn.L1Loss()

        optimzr = optim.Adam(model.parameters(), lr=lr)
        model.train()

        # DataLoader cho mini-batch
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if global_weights is not None and self.federated_type == "FedProx":
            # Tạo bản copy và chuyển về device
            global_weights_copy = {
                k: v.clone().detach().to(self.device)
                for k, v in global_weights.items()
            }
        else:
            global_weights_copy = None


        for ep in range(local_epochs):
            epoch_loss = 0.0
            epoch_mae = 0.0
            epoch_mse = 0.0
            num_batches = 0
            for batch in loader:
                xb = batch[0]

                optimzr.zero_grad()
                pred = model(xb)

                if isinstance(model, VAELSTM):
                    recon_x, mu_vae, logvar = pred
                    recon_loss = criterion(recon_x, xb)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu_vae.pow(2) - logvar.exp())
                    loss = recon_loss + beta * kl_loss
                    mse_v = recon_loss.item()
                    mae_v = l1(recon_x, xb).item()

                elif isinstance(model, VAE):
                    recon_x, mu_vae, logvar = pred
                    recon_loss = criterion(recon_x, xb)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu_vae.pow(2) - logvar.exp())
                    loss = recon_loss + beta * kl_loss
                    mse_v = recon_loss.item()
                    mae_v = l1(recon_x, xb).item()

                elif isinstance(model, AELSTM) or isinstance(model, AECNN):
                    recon = pred
                    loss = criterion(recon, xb)
                    mse_v = loss.item()
                    mae_v = l1(recon, xb).item()

                elif isinstance(model, LSTMWithIForest):
                    feats = pred
                    loss = (feats ** 2).mean()
                    mse_v = loss.item()
                    mae_v = torch.mean(torch.abs(feats)).item()


                elif isinstance(model, LSTMModel):
                    pred_shifted = pred[:, :-1]
                    target_shifted = xb[:, 1:]
                    loss = criterion(pred_shifted, target_shifted)
                    mse_v = loss.item()
                    mae_v = l1(pred_shifted, target_shifted).item()

                # FedProx
                if self.federated_type == "FedProx" and global_weights_copy is not None:
                    prox_loss = fedprox_loss(
                        model,
                        global_weights_copy,
                        mu=mu,
                        device=self.device
                    )
                    loss = loss + prox_loss

                loss.backward()

                optimzr.step()

                epoch_loss += loss.item()
                epoch_mae += mae_v
                epoch_mse += mse_v
                num_batches += 1

            epoch_loss /= num_batches
            epoch_mae /= num_batches
            epoch_mse /= num_batches

        metrics = {
            "train_loss": epoch_loss,
            "train_mae": epoch_mae,
            "train_mse": epoch_mse
        }
        return model, metrics



    def get_local_update(self, global_weights=None, current_round=0):
        if self._get_input_dim() == 0:
            return None, 0, {"train_loss": 0.0, "train_mae": 0.0, "train_mse": 0.0}

        X_train, _, _ = self.split_data()
        local_model = self.build_model()

        if local_model is None:
            return None, 0, {"train_loss": 0.0, "train_mae": 0.0, "train_mse": 0.0}

        if global_weights is not None:
            local_model.load_state_dict(global_weights, strict=False)

        trained_model, metrics = self.train_model(
            local_model,
            X_train,
            global_weights=global_weights,
            current_round=current_round
        )

        return trained_model.state_dict(), len(X_train), metrics


class FedServerLogic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_type = cfg["model"]["type"]
        self.window_len = cfg["model"]["window_len"]
        self.overlap = cfg["model"]["overlap"]
        self.federated_type = cfg["federated"]["type"]
        self.full_data_path = "../data/data_hl19_real.csv" ###self.full_data_path = "../data/data_hl19_full.csv" (full: giả lập, real)

        print(f"[Serverlogic] Đang load Full Dataset từ: {self.full_data_path}")

        try:
            full_dataset_loader = FedLocalTrain(
                client_id="global",
                data_path=self.full_data_path,
                model_type=self.model_type,
                window_len=self.window_len,
                overlap=self.overlap,
                federated_type=self.federated_type
            )

            inp = full_dataset_loader._get_input_dim()
            X_train_global,X_val_global, X_test_global = full_dataset_loader.split_data()

            self.server_eval_data = X_val_global
            self.server_train_data = X_train_global
            print(f"[Serverlogic] Đã tạo Global Test Set: {len(self.server_eval_data)} mẫu.")

        except Exception as e:
            print(f"[Serverlogic]LỖI: Không load được file full data ({e}).")
            print("[Serverlogic] Server sẽ không thể đánh giá chính xác Global Model.")
            inp = 1
            self.server_eval_data = None

        if inp == 0: inp = 1
        self.input_dim = inp

        self.global_model = self._build_global_model(inp)
        print(f"[ServerLogic] Global Model initialized (input_dim={inp})")

    def _build_global_model(self, inp):
        if self.model_type == "lstm":
            return LSTMModel(input_dim=inp)
        if self.model_type == "ae_lstm":
            return AELSTM(input_dim=inp)
        if self.model_type == "iforest_lstm":
            return LSTMWithIForest(input_dim=inp)
        if self.model_type == "vae_lstm":
            return VAELSTM(input_dim=inp)
        if self.model_type == "vae":
            return VAE(input_dim=inp * self.window_len)
        if self.model_type == "ae_cnn":
            return AECNN(input_dim=inp, seq_len=self.window_len)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def get_global_state(self):
        return self.global_model.state_dict()

    def set_global_state(self, state_dict):
        self.global_model.load_state_dict(state_dict, strict=False)

    '''
    def evaluate_global(self,X_train=None, X_test=None):
        if X_test is None:
            X_test = self.server_eval_data

        if X_train is None:
            X_train = self.server_train_data

        if X_test is None or len(X_test) == 0:
            return {"loss": 0.0, "mse": 0.0, "mae": 0.0}

        batch_size = self.cfg["training"]["batch_size"]

        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_train = torch.tensor(X_train, dtype=torch.float32)

        dataset_train = torch.utils.data.TensorDataset(X_train)
        dataset = torch.utils.data.TensorDataset(X_test)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)


        #model = self.global_model
        model = copy.deepcopy(self.global_model)

        if self.federated_type == "FedBN":

            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.momentum = 1.0 # Set momentum = 1.0

            model.train()  # Bật chế độ train để BN tính toán mean/var
            with torch.no_grad():
                # Chạy qua toàn bộ (hoặc một phần) dữ liệu test để update BN stats
                for batch in train_loader:
                    xb = batch[0]
                    _ = model(xb)  # Forward pass để cập nhật running_mean/var

        model.eval()  # Sau khi calibrate xong thì chuyển về eval để test

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        beta = self.cfg["training"].get("beta", 0.001)

        with torch.no_grad():
            for batch in test_loader:
                xb = batch[0]
                loss, mse, mae, _ = compute_loss(model, xb, beta=beta)

                if isinstance(loss, torch.Tensor):
                    loss = loss.item()

                bs = xb.size(0)
                total_loss += loss * bs
                total_mse += mse * bs
                total_mae += mae * bs
                total_samples += bs

        total_loss /= total_samples
        total_mse /= total_samples
        total_mae /= total_samples

        return {
            "loss": total_loss,
            "mse": total_mse,
            "mae": total_mae
        }
'''
    def evaluate_global(self, X_val=None, X_test=None):
        if X_val is None:
            X_val = self.server_eval_data
        if X_val is None or len(X_val) == 0:
            return {"loss": 0.0, "mse": 0.0, "mae": 0.0}

        batch_size = self.cfg["training"]["batch_size"]

        X = torch.tensor(X_val, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = self.global_model
        model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        beta = self.cfg["training"].get("beta", 0.001)

        with torch.no_grad():
            for batch in loader:
                xb = batch[0]
                loss, mse, mae, _ = compute_loss(model, xb, beta=beta)

                if isinstance(loss, torch.Tensor): loss = loss.item()
                if isinstance(mse, torch.Tensor): mse = mse.item()
                if isinstance(mae, torch.Tensor): mae = mae.item()

                bs = xb.size(0)
                total_loss += loss * bs
                total_mse += mse * bs
                total_mae += mae * bs
                total_samples += bs

        if total_samples == 0:
            return {"loss": 0.0, "mse": 0.0, "mae": 0.0}

        total_loss /= total_samples
        total_mse /= total_samples
        total_mae /= total_samples

        return {
            "loss": total_loss,
            "mse": total_mse,
            "mae": total_mae
        }
