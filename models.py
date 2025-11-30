import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class AELSTM(nn.Module):
    def __init__(self, input_dim, enc_hidden=64, dec_hidden=64, bottleneck=32, enc_layers=1, dec_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.bottleneck = bottleneck

        # Encode
        self.encoder_lstm = nn.LSTM(
            input_dim,
            enc_hidden,
            enc_layers,
            batch_first=True
        )
        self.encoder_fc = nn.Linear(enc_hidden, bottleneck)

        #Decode
        self.decoder_fc = nn.Linear(bottleneck, dec_hidden)

        # LSTM decoder (input_size = dec_hidden, hidden_size = dec_hidden)
        self.decoder_lstm = nn.LSTM(
            dec_hidden,
            dec_hidden,
            dec_layers,
            batch_first=True
        )
        self.output_fc = nn.Linear(dec_hidden, input_dim)

    def forward(self, x):
        batch, seq_len, _ = x.size()

        enc_out, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]                 # (batch, enc_hidden)
        z = self.encoder_fc(h_last)      # (batch, bottleneck)

        dec_in = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_in)  # (batch, seq_len, dec_hidden)

        recon = self.output_fc(dec_out)         # (batch, seq_len, input_dim)
        return recon


class LSTMWithIForest(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, proj_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, proj_dim)
        self.iforest = IsolationForest(n_estimators=200)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

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

        # Mean và Log-Variance của phân phối tiềm ẩn
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 2. DECODER (LSTM)
        # Bắt đầu từ không gian tiềm ẩn (latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder_lstm = nn.LSTM(
            hidden_dim,  # Input size là hidden_dim sau khi giải nén
            hidden_dim,
            layers,
            batch_first=True
        )

        # Trả về kích thước đầu vào (input_dim)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # ENCODER
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]  # Lấy hidden state cuối cùng của lớp cuối cùng (batch, hidden_dim)

        # Tính toán Mean (mu) và Log-Variance (logvar)
        mu = self.fc_mean(h_last)
        logvar = self.fc_logvar(h_last)

        # Lấy mẫu từ phân phối tiềm ẩn
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)

        # DECODER
        # 1. Mở rộng z thành chuỗi có kích thước seq_len
        dec_in = self.fc_decoder_input(z).unsqueeze(1)  # (batch, 1, hidden_dim)
        dec_in = dec_in.repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)

        # 2. Giải mã bằng LSTM
        dec_out, _ = self.decoder_lstm(dec_in)  # (batch, seq_len, hidden_dim)

        # 3. Tái tạo đầu vào
        recon_x = self.output_fc(dec_out)  # (batch, seq_len, input_dim)

        # Trả về kết quả tái tạo, Mean và Log-Variance (dùng cho Loss)
        return recon_x, mu, logvar
