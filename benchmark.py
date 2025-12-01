import argparse
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from fedml_logic import FedLocalTrain, compute_loss

cfg = yaml.safe_load(open("config.yml", "r"))


def load_global_logs(client_id):
    json_path = f"outputs/client_{client_id}_metrics.json"
    if not os.path.exists(json_path):
        print(f"[WARN] Missing {json_path}")
        return None
    logs = json.load(open(json_path))
    df = pd.DataFrame(logs).sort_values("round")
    return df


def run_benchmark(client_id, global_weights_path):

    print("\nBENCHMARK CLIENT", client_id, "\n")

    client = FedLocalTrain(
        client_id=client_id,
        data_path=f"data/data_hl19_node_{client_id}.csv",
        model_type=cfg["model"]["type"],
        window_len=cfg["model"]["window_len"],
        overlap=cfg["model"]["overlap"]
    )

    X_train, X_val, X_test = client.split_data()


    global_model = client.build_model()
    try:
        g_weights = torch.load(global_weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        g_weights = torch.load(global_weights_path, map_location="cpu")

    global_model.load_state_dict(g_weights)
    global_model.eval()

    # Train local model
    local_model = client.build_model()
    epochs = cfg["federated"]["rounds"]

    print(f"Training LOCAL model for {epochs} epochs...")
    local_hist = []

    for ep in range(1, epochs + 1):
        local_model, met = client.compute_local_train_step(local_model, X_train)
        local_hist.append({
            "epoch": ep,
            "loss": met["train_loss"],
            "mae": met["train_mae"],
            "mse": met["train_mse"]
        })

    # Evaluate Local vs Global
    metrics_local  = client.compute_metrics(local_model,  X_test, X_train)
    metrics_global = client.compute_metrics(global_model, X_test, X_train)

    df_global = load_global_logs(client_id)

# plot
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(1, 2, 1)

    e = [x["epoch"] for x in local_hist]
    mae_local = [x["mae"] for x in local_hist]
    mse_local = [x["mse"] for x in local_hist]

    ax1.plot(e, mae_local, color="blue", label="Local MAE (train)")
    ax1.plot(e, mse_local, color="green", label="Local MSE (train)")

    if df_global is not None:
        ax1.plot(df_global["round"], df_global["mae"], linestyle="--",
                 color="red", label="Global MAE (FL)")
        ax1.plot(df_global["round"], df_global["mse"], linestyle="--",
                 color="orange", label="Global MSE (FL)")

    ax1.set_xlabel("Epoch / Round")
    ax1.set_ylabel("Value")
    ax1.set_title(f"Client {client_id} – MAE/MSE Local vs Global")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    labels = ["MAE", "MSE"]
    local_vals = [metrics_local["mae"], metrics_local["mse"]]
    global_vals = [metrics_global["mae"], metrics_global["mse"]]

    x = np.arange(len(labels))
    w = 0.35

    ax2.bar(x - w / 2, local_vals,  w, label="Local",  color="#4A90E2")
    ax2.bar(x + w / 2, global_vals, w, label="Global", color="#E24A4A")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Value")
    ax2.set_title(f"Client {client_id} – Test MAE/MSE")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Save
    plt.tight_layout()
    out_all = f"outputs/benchmark_client_{client_id}_global_local.png"
    plt.savefig(out_all, dpi=150)
    plt.close()

    print(" Saved figure:", out_all)


    print("\nRESULT SUMMARY")
    print(f"Local Test MAE:   {metrics_local['mae']:.6f}")
    print(f"Global Test MAE:  {metrics_global['mae']:.6f}")
    print(f"Local Test MSE:   {metrics_local['mse']:.6f}")
    print(f"Global Test MSE:  {metrics_global['mse']:.6f}")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--global", dest="gw", default="outputs/global_model_final.pth")
    args = parser.parse_args()

    run_benchmark(args.id, args.gw)
