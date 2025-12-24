import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

'''
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        h_last = self.bn_lstm(h_last)
        return self.fc(h_last)
'''


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            layers,
            batch_first=True
        )
        self.bn_lstm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):

        batch_size, seq_len, _ = x.size()

        out, _ = self.lstm(x)  # (Batch, Seq_Len, Hidden_Dim)
        out = out.permute(0, 2, 1)  # (Batch, Hidden_Dim, Seq_Len)
        out = self.bn_lstm(out)
        out = out.permute(0, 2, 1)  # (Batch, Seq_Len, Hidden_Dim)
        predictions = self.fc(out)  # (Batch, Seq_Len, Input_Dim)
        return predictions


class AELSTM(nn.Module):
    def __init__(self, input_dim, enc_hidden=64, dec_hidden=64, bottleneck=32, enc_layers=1, dec_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.bottleneck = bottleneck
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim,
            enc_hidden,
            enc_layers,
            batch_first=True
        )
        self.bn_encoder = nn.BatchNorm1d(enc_hidden)
        self.encoder_fc = nn.Linear(enc_hidden, bottleneck)
        self.bn_bottleneck = nn.BatchNorm1d(bottleneck)
        # Decoder
        self.decoder_fc = nn.Linear(bottleneck, dec_hidden)
        self.bn_decoder_input = nn.BatchNorm1d(dec_hidden)
        self.decoder_lstm = nn.LSTM(
            dec_hidden,
            dec_hidden,
            dec_layers,
            batch_first=True
        )

        self.output_fc = nn.Linear(dec_hidden, input_dim)

    def forward(self, x):
        batch, seq_len, _ = x.size()

        # Encoding
        enc_out, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]  # (batch, enc_hidden)

        # [FEDBN] Áp dụng Batch Norm
        h_last = self.bn_encoder(h_last)
        z = self.encoder_fc(h_last)  # (batch, bottleneck)
        # [FEDBN] Áp dụng Batch Norm
        z = self.bn_bottleneck(z)
        # Decoding
        dec_in_flat = self.decoder_fc(z)  # (batch, dec_hidden)
        # [FEDBN] Áp dụng Batch Norm
        dec_in_flat = self.bn_decoder_input(dec_in_flat)
        dec_in = dec_in_flat.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_in)  # (batch, seq_len, dec_hidden)
        recon = self.output_fc(dec_out)  # (batch, seq_len, input_dim)
        return recon


class LSTMWithIForest(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, proj_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # [FEDBN] Batch Norm sau LSTM
        self.bn_lstm = nn.BatchNorm1d(hidden_dim)

        self.fc = nn.Linear(hidden_dim, proj_dim)

        # [FEDBN] Batch Norm sau projection
        self.bn_proj = nn.BatchNorm1d(proj_dim)

        self.iforest = IsolationForest(n_estimators=200)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]  # (batch, hidden_dim)

        # [FEDBN] Áp dụng Batch Norm
        h_last = self.bn_lstm(h_last)

        proj = self.fc(h_last)  # (batch, proj_dim)

        # [FEDBN] Áp dụng Batch Norm
        proj = self.bn_proj(proj)

        return proj

    def fit_iforest(self, feats):
        self.iforest.fit(feats.detach().cpu().numpy())

    def anomaly_score(self, feats):
        return -self.iforest.score_samples(feats.detach().cpu().numpy())


class VAELSTM(nn.Module):
    def __init__(self, input_dim, seq_len=6, hidden_dim=64, latent_dim=16, layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.layers = layers

        # 1. ENCODER (LSTM)
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            layers,
            batch_first=True
        )

        # [FEDBN] Batch Norm sau Encoder LSTM
        self.bn_encoder = nn.BatchNorm1d(hidden_dim)

        # Mean và Log-Variance của phân phối tiềm ẩn
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 2. DECODER (LSTM)
        self.fc_decoder_input = nn.Linear(latent_dim, hidden_dim)

        # [FEDBN] Batch Norm sau khi project từ latent
        self.bn_decoder_input = nn.BatchNorm1d(hidden_dim)

        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            layers,
            batch_first=True
        )

        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]  # (batch, hidden_dim)

        h_last = self.bn_encoder(h_last)
        mu = self.fc_mean(h_last)
        logvar = self.fc_logvar(h_last)
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)

        # DECODER
        dec_in_flat = self.fc_decoder_input(z)  # (batch, hidden_dim)

        dec_in_flat = self.bn_decoder_input(dec_in_flat)
        dec_in = dec_in_flat.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_in)
        recon_x = self.output_fc(dec_out)

        return recon_x, mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()

        # --- 1. ENCODER ---
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # [FEDBN] Batch Norm sau lớp Linear đầu tiên
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- 2. DECODER ---
        self.fc2 = nn.Linear(latent_dim, hidden_dim)

        # [FEDBN] Batch Norm sau khi giải nén từ Latent space
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """
        Chiêu thức Reparameterization để cho phép backpropagation qua tầng lấy mẫu ngẫu nhiên
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Flatten input nếu cần: (batch, seq, feat) -> (batch, seq*feat)
        original_shape = x.shape
        if x.ndim == 3:
            batch_size, seq_len, feat_dim = x.shape
            x = x.reshape(batch_size, seq_len * feat_dim)

        # --- ENCODE ---
        h = self.fc1(x)
        h = self.bn1(h)  # [FEDBN] Áp dụng Batch Norm
        h = F.relu(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        # --- DECODE ---
        dec_h = self.fc2(z)
        dec_h = self.bn2(dec_h)  # [FEDBN] Áp dụng Batch Norm
        dec_h = F.relu(dec_h)

        recon_x = self.fc3(dec_h)

        # Reshape lại về kích thước ban đầu nếu input là 3D
        if len(original_shape) == 3:
            recon_x = recon_x.reshape(original_shape)

        return recon_x, mu, logvar


class AECNN(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_channels=[32, 16], kernel_size=3, stride=2):
        super(AECNN, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        # --- ENCODER ---
        # Input: (Batch, Input_Dim, Seq_Len)
        self.enc_conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_channels[0],
                                   kernel_size=kernel_size, stride=stride, padding=1)
        self.enc_bn1 = nn.BatchNorm1d(hidden_channels[0])

        self.enc_conv2 = nn.Conv1d(in_channels=hidden_channels[0], out_channels=hidden_channels[1],
                                   kernel_size=kernel_size, stride=stride, padding=1)
        self.enc_bn2 = nn.BatchNorm1d(hidden_channels[1])

        # --- DECODER ---
        # ConvTranspose1d để upsample
        self.dec_conv1 = nn.ConvTranspose1d(in_channels=hidden_channels[1], out_channels=hidden_channels[0],
                                            kernel_size=kernel_size, stride=stride, padding=1)
        self.dec_bn1 = nn.BatchNorm1d(hidden_channels[0])

        # Layer cuối đưa về đúng số features ban đầu (Input_Dim)
        self.dec_conv2 = nn.ConvTranspose1d(in_channels=hidden_channels[0], out_channels=input_dim,
                                            kernel_size=kernel_size, stride=stride, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape gốc: (Batch, Seq_Len, Features)
        # Conv1d cần: (Batch, Features, Seq_Len)

        # 1. Transpose Input
        if x.shape[1] != self.enc_conv1.in_channels and x.shape[2] == self.enc_conv1.in_channels:
            x = x.transpose(1, 2)

            # Lưu lại kích thước gốc để resize lúc cuối
        orig_size = x.size(2)

        # --- Encode ---
        e1 = self.relu(self.enc_bn1(self.enc_conv1(x)))
        e2 = self.relu(self.enc_bn2(self.enc_conv2(e1)))

        # --- Decode ---
        d1 = self.relu(self.dec_bn1(self.dec_conv1(e2)))
        out = self.dec_conv2(d1)

        if out.size(2) != orig_size:
            out = F.interpolate(out, size=orig_size, mode='linear', align_corners=False)

        out = out.transpose(1, 2)

        return out