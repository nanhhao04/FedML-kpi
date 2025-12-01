import io
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from preprocess import load_and_scale_data, create_sequence_with_date, split_train_val_test
from models import LSTMModel, AELSTM, LSTMWithIForest, VAELSTM

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


def aggregate_weights(local_updates):
    """Aggregate local_updates which is a list of tuples (state_dict, n_samples).

    Returns a new aggregated state_dict (weighted average) or None if empty.
    """
    if not local_updates:
        return None

    sizes = [n for _, n in local_updates]
    total = float(sum(sizes))
    first = local_updates[0][0]

    new_global = {}
    for key in first.keys():
        new_global[key] = torch.zeros_like(first[key])
        for w, n in local_updates:
            new_global[key] += w[key] * (float(n) / total)

    return new_global


def compute_loss(model, X, beta=None):
    """Compute unified loss, mse, mae and per-sample scores for different model types.

    Returns: (loss_tensor, mse_value, mae_value, scores_tensor)
    - loss_tensor: scalar torch tensor used for backward (already on cpu)
    - mse_value, mae_value: python floats
    - scores_tensor: 1D torch tensor (per-sample anomaly / error scores)
    """
    if beta is None:
        beta = cfg["training"].get("beta", 0.001)

    criterion = nn.MSELoss()
    l1 = nn.L1Loss()

    model.eval()  # safe to call; training loops will set train() explicitly
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
        # per-sample: mean squared error across sequence & features
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

    # LSTM features for IForest
    if isinstance(model, LSTMWithIForest):
        feats = pred
        loss = (feats ** 2).mean()
        mse = loss.item()
        mae = torch.mean(torch.abs(feats)).item()
        # per-sample anomaly score; leave as tensor (model may convert later)
        scores = feats.norm(dim=1)
        return loss, mse, mae, scores

    # Plain LSTM (sequence -> next-step prediction or multi-step)
    # pred may be (batch, seq, feat) or (batch, feat)
    pred_proc = pred
    if pred_proc.ndim == 3:
        pred_proc = pred_proc[:, -1, :]

    if pred_proc.shape[-1] == input_dim:
        tgt = X[:, -1, :]
    else:
        tgt = X[:, -1, 0].unsqueeze(1)

    loss = criterion(pred_proc, tgt)
    mse = loss.item()
    mae = l1(pred_proc, tgt).item()
    scores = ((pred_proc - tgt) ** 2).mean(dim=1)

    return loss, mse, mae, scores


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

        # store optimizer for local benchmark
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

        raise ValueError(f"Unknown model_type: {self.model_type}")

    def split_data(self):
        return split_train_val_test(self.X, train_ratio=0.7, val_ratio=0.15)

    def train_model(self, model, X_train, current_round=0):
        """Train model for given number of epochs (currently 1).
        REMOVED: Learning Rate Decay (Always use fixed LR from config).
        """
        if X_train is None or len(X_train) == 0:
            return model, {"train_loss": 0.0, "train_mae": 0.0, "train_mse": 0.0}

        epochs = 1
        # Lấy trực tiếp Learning Rate từ config và giữ nguyên
        lr = cfg["training"]["lr"]
        beta = cfg["training"].get("beta", 0.001)

        # --- ĐÃ XÓA LOGIC DECAY ---
        # decay_rate = 0.9
        # decay_step = 10
        # current_lr = ... (Không tính nữa)

        criterion = nn.MSELoss()
        l1 = nn.L1Loss()

        # Luôn dùng lr cố định
        optimzr = optim.Adam(model.parameters(), lr=lr)
        model.train()

        X = torch.tensor(X_train, dtype=torch.float32)

        last_loss = 0.0
        last_mae = 0.0
        last_mse = 0.0

        for _ in range(epochs):
            optimzr.zero_grad()

            # For compute_loss we need model(X) in training mode and allow grad
            pred = model(X)

            # We reuse the same logic as compute_loss but without torch.no_grad
            # VAE
            if isinstance(model, VAELSTM):
                recon_x, mu, logvar = pred
                recon_loss = criterion(recon_x, X)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss

                mse_v = recon_loss.item()
                mae_v = l1(recon_x, X).item()

            elif isinstance(model, AELSTM):
                recon = pred
                loss = criterion(recon, X)
                mse_v = loss.item()
                mae_v = l1(recon, X).item()

            elif isinstance(model, LSTMWithIForest):
                feats = pred
                loss = (feats ** 2).mean()
                mse_v = loss.item()
                mae_v = torch.mean(torch.abs(feats)).item()

            else:
                pred_proc = pred
                if pred_proc.ndim == 3:
                    pred_proc = pred_proc[:, -1, :]

                input_dim = X.shape[-1]
                if pred_proc.shape[-1] == input_dim:
                    target = X[:, -1, :]
                else:
                    target = X[:, -1, 0].unsqueeze(1)

                loss = criterion(pred_proc, target)
                mse_v = loss.item()
                mae_v = l1(pred_proc, target).item()

            loss.backward()
            optimzr.step()

            last_loss = loss.item()
            last_mse = mse_v
            last_mae = mae_v

        metrics = {"train_loss": last_loss, "train_mae": last_mae, "train_mse": last_mse}
        return model, metrics

    def get_local_update(self, global_weights=None, current_round=0):
        if self._get_input_dim() == 0:
            return None, 0, {"train_loss": 0.0, "train_mae": 0.0, "train_mse": 0.0}

        X_train, _, _ = self.split_data()
        local_model = self.build_model()

        if local_model is None:
            return None, 0, {"train_loss": 0.0, "train_mae": 0.0, "train_mse": 0.0}

        # Load global weights if provided
        if global_weights is not None:
            local_model.load_state_dict(global_weights)

        trained_model, metrics = self.train_model(local_model, X_train, current_round=current_round)

        return trained_model.state_dict(), len(X_train), metrics

    def compute_local_train_step(self, model, X_train):
        """One epoch/step for local benchmark. Uses persistent optimizer (no scheduler).
        Returns updated model and metrics dict.
        """
        if X_train is None or len(X_train) == 0:
            return model, {"train_loss": 0.0, "train_mse": 0.0, "train_mae": 0.0}

        lr = cfg["training"]["lr"]
        beta = cfg["training"].get("beta", 0.001)

        if self.local_optimizer is None:
            self.local_optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        X = torch.tensor(X_train, dtype=torch.float32)

        self.local_optimizer.zero_grad()

        pred = model(X)

        # compute loss consistent with train_model
        if isinstance(model, VAELSTM):
            recon_x, mu, logvar = pred
            recon_loss = nn.MSELoss()(recon_x, X)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
            mse_v = recon_loss.item()
            mae_v = nn.L1Loss()(recon_x, X).item()

        elif isinstance(model, AELSTM):
            loss = nn.MSELoss()(pred, X)
            mse_v = loss.item()
            mae_v = nn.L1Loss()(pred, X).item()

        elif isinstance(model, LSTMWithIForest):
            feats = pred
            loss = (feats ** 2).mean()
            mse_v = loss.item()
            mae_v = torch.mean(torch.abs(feats)).item()

        else:
            pred_proc = pred
            if pred_proc.ndim == 3:
                pred_proc = pred_proc[:, -1, :]

            input_dim = X.shape[-1]
            if pred_proc.shape[-1] == input_dim:
                target = X[:, -1, :]
            else:
                target = X[:, -1, 0].unsqueeze(1)

            loss = nn.MSELoss()(pred_proc, target)
            mse_v = loss.item()
            mae_v = nn.L1Loss()(pred_proc, target).item()

        loss.backward()
        self.local_optimizer.step()

        return model, {"train_loss": loss.item(), "train_mse": mse_v, "train_mae": mae_v}

    def compute_metrics(self, model, X_test, X_train=None):
        """Evaluate model on X_test. If model is LSTMWithIForest and X_train is provided,
        fit isolation forest internal state before scoring.
        Returns dict {mse, mae, scores}
        """
        if X_test is None or len(X_test) == 0:
            return {"mse": 0.0, "mae": 0.0, "scores": np.array([])}

        model.eval()
        X = torch.tensor(X_test, dtype=torch.float32)

        # For LSTMWithIForest, model may need to fit IForest on train features
        if isinstance(model, LSTMWithIForest) and X_train is not None:
            with torch.no_grad():
                feats_tr = model(torch.tensor(X_train, dtype=torch.float32))
            model.fit_iforest(feats_tr)

        # Use compute_loss logic but allow grad disabled
        loss, mse, mae, scores = compute_loss(model, X, beta=cfg["training"].get("beta", 0.001))

        return {"mse": mse, "mae": mae, "scores": scores.cpu().numpy()}


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
            overlap=self.overlap,
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
        raise ValueError(f"Unknown model type: {self.model_type}")

    def get_global_state(self):
        return self.global_model.state_dict()

    def set_global_state(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def evaluate_global(self, X_test):
        """Evaluate the global model on X_test using the unified compute_loss."""
        if X_test is None or len(X_test) == 0:
            return {"mse": 0.0, "mae": 0.0}

        X = torch.tensor(X_test, dtype=torch.float32)
        _, mse, mae, _ = compute_loss(self.global_model, X, beta=cfg["training"].get("beta", 0.001))
        return {"mse": mse, "mae": mae}


# End of file
