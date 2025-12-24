# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = "../outputs_iid_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"      FULL NON-IID ANALYSIS TOOL (PCA + JS + WD + KS + HD + SCORE)")
print("=" * 60)
print(f"-> Output directory: {OUTPUT_DIR}/")


def load_data(client_id):
    path = f"../data/data_hl19_node_{client_id}.csv"
    try:
        df = pd.read_csv(path)

        drop_cols = ['date', 'NODE_ID', 'timestamp', 'label', 'anomaly']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        df = df.select_dtypes(include=[np.number])

        return df
    except FileNotFoundError:
        return None


print("-> [1/7] Đang load dữ liệu 3 Node...")
df1 = load_data(1)
df2 = load_data(2)
df3 = load_data(3)

if df1 is None or df2 is None or df3 is None:
    print("ERROR: Thiếu file data/data_hl19_node_X.csv")
    exit()

features = df1.columns.tolist()
print(f"   ✔ Tổng cộng {len(features)} features dạng số.")


print("-> [2/7] Global MinMax Scaling...")

combined_raw = pd.concat([df1, df2, df3], axis=0)
scaler = MinMaxScaler()
scaler.fit(combined_raw)

df1_scaled = pd.DataFrame(scaler.transform(df1), columns=features)
df2_scaled = pd.DataFrame(scaler.transform(df2), columns=features)
df3_scaled = pd.DataFrame(scaler.transform(df3), columns=features)

def calculate_js_distance(p, q, bins=50):
    range_min = min(p.min(), q.min())
    range_max = max(p.max(), q.max())

    p_hist, _ = np.histogram(p, bins=bins, range=(range_min, range_max), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(range_min, range_max), density=True)

    p_hist = np.where(p_hist == 0, 1e-10, p_hist)
    q_hist = np.where(q_hist == 0, 1e-10, q_hist)

    return jensenshannon(p_hist, q_hist)


def hellinger_distance(p, q, bins=50):
    # build common histogram bins
    rmin = min(p.min(), q.min())
    rmax = max(p.max(), q.max())
    hp, _ = np.histogram(p, bins=bins, range=(rmin, rmax), density=True)
    hq, _ = np.histogram(q, bins=bins, range=(rmin, rmax), density=True)

    # normalize to probability vectors
    hp = hp / (hp.sum() + 1e-12)
    hq = hq / (hq.sum() + 1e-12)

    return np.sqrt(0.5 * np.sum((np.sqrt(hp + 1e-12) - np.sqrt(hq + 1e-12)) ** 2))


print("-> [3/7] Tính toán Wasserstein, KS Test, Jensen–Shannon, Hellinger...")

rows = []
pair_scores = {"12": [], "13": [], "23": []}

for col in features:
    d1 = df1_scaled[col].dropna()
    d2 = df2_scaled[col].dropna()
    d3 = df3_scaled[col].dropna()

    # Wasserstein (on scaled data 0-1)
    ws12 = wasserstein_distance(d1, d2)
    ws13 = wasserstein_distance(d1, d3)
    ws23 = wasserstein_distance(d2, d3)
    avg_ws = np.mean([ws12, ws13, ws23])

    # KS test p-values (take min p-value as conservative)
    ks12 = ks_2samp(d1, d2).pvalue
    ks13 = ks_2samp(d1, d3).pvalue
    ks23 = ks_2samp(d2, d3).pvalue
    min_pval = min(ks12, ks13, ks23)

    # JS distance (shape)
    js12 = calculate_js_distance(d1, d2)
    js13 = calculate_js_distance(d1, d3)
    js23 = calculate_js_distance(d2, d3)
    avg_js = np.mean([js12, js13, js23])

    # Hellinger distance
    hd12 = hellinger_distance(d1, d2)
    hd13 = hellinger_distance(d1, d3)
    hd23 = hellinger_distance(d2, d3)
    avg_hd = np.mean([hd12, hd13, hd23])

    # Non-IID Score cho từng cặp (Wasserstein / JS / Hellinger / KS)
    # note: KS contributes via (1 - pval) so lower pval -> larger contribution
    pair_scores["12"].append(0.45 * ws12 + 0.25 * js12 + 0.2 * hd12 + 0.1 * (1 - ks12))
    pair_scores["13"].append(0.45 * ws13 + 0.25 * js13 + 0.2 * hd13 + 0.1 * (1 - ks13))
    pair_scores["23"].append(0.45 * ws23 + 0.25 * js23 + 0.2 * hd23 + 0.1 * (1 - ks23))

    rows.append({
        "Feature": col,
        "Wasserstein": avg_ws,
        "JS_Dist": avg_js,
        "Hellinger": avg_hd,
        "KS_Pval": min_pval
    })

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(f"{OUTPUT_DIR}/metrics_details.csv", index=False)
print(f"   ✔ Xuất metrics_details.csv")


print("-> [3.5/7] Tính Non-IID Score tổng hợp (pairwise average)...")

summary_scores = {
    "Node1 vs Node2": np.mean(pair_scores["12"]),
    "Node1 vs Node3": np.mean(pair_scores["13"]),
    "Node2 vs Node3": np.mean(pair_scores["23"]),
}

pd.DataFrame.from_dict(summary_scores, orient='index', columns=["NonIID_Score"]) \
    .to_csv(f"{OUTPUT_DIR}/non_iid_pair_score.csv")

print(f"   ✔ Xuất non_iid_pair_score.csv")



print("-> [4/7] Vẽ PCA + KDE...")

n_samp = 500
s1 = df1_scaled.sample(min(n_samp, len(df1)))
s2 = df2_scaled.sample(min(n_samp, len(df2)))
s3 = df3_scaled.sample(min(n_samp, len(df3)))

X_pca_input = np.vstack([s1, s2, s3])
labels = ['Node1'] * len(s1) + ['Node2'] * len(s2) + ['Node3'] * len(s3)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_pca_input)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, s=60, alpha=0.7)
plt.title("PCA View (Scaled Data)")
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/1_pca_structure.png")
plt.close()

# KDE plots for top features
top3_wd = metrics_df.sort_values("Wasserstein", ascending=False).head(3)["Feature"].tolist()
top3_hd = metrics_df.sort_values("Hellinger", ascending=False).head(3)["Feature"].tolist()

plt.figure(figsize=(18, 5))
for i, col in enumerate(top3_wd):
    plt.subplot(1, 3, i + 1)
    sns.kdeplot(df1[col], label="Node1", fill=True, alpha=0.3)
    sns.kdeplot(df2[col], label="Node2", fill=True, alpha=0.3)
    sns.kdeplot(df3[col], label="Node3", fill=True, alpha=0.3)
    plt.title(f"Top WD Feature: {col}")
    plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/2_kde_top_wd.png")
plt.close()

plt.figure(figsize=(18, 5))
for i, col in enumerate(top3_hd):
    plt.subplot(1, 3, i + 1)
    sns.kdeplot(df1[col], label="Node1", fill=True, alpha=0.3)
    sns.kdeplot(df2[col], label="Node2", fill=True, alpha=0.3)
    sns.kdeplot(df3[col], label="Node3", fill=True, alpha=0.3)
    plt.title(f"Top HD Feature: {col}")
    plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/3_kde_top_hd.png")
plt.close()


print("-> [4.5/7] Vẽ Non-IID Heatmap...")

matrix = np.array([
    [0, summary_scores["Node1 vs Node2"], summary_scores["Node1 vs Node3"]],
    [summary_scores["Node1 vs Node2"], 0, summary_scores["Node2 vs Node3"]],
    [summary_scores["Node1 vs Node3"], summary_scores["Node2 vs Node3"], 0]
])

plt.figure(figsize=(6, 4))
sns.heatmap(matrix, annot=True, cmap="Reds",
            xticklabels=["Node1", "Node2", "Node3"],
            yticklabels=["Node1", "Node2", "Node3"])
plt.title("Non-IID Heatmap (Overall Differences)")
plt.savefig(f"{OUTPUT_DIR}/0_noniid_heatmap.png")
plt.close()


print("-> [5/7] Đánh giá mức độ hòa trộn bằng Silhouette + DB Index + Calinski-Harabasz...")

X_all = np.vstack([df1_scaled, df2_scaled, df3_scaled])
cluster_labels = np.array(
    [0] * len(df1_scaled) +
    [1] * len(df2_scaled) +
    [2] * len(df3_scaled)
)

sil = silhouette_score(X_all, cluster_labels)
db = davies_bouldin_score(X_all, cluster_labels)
ch = calinski_harabasz_score(X_all, cluster_labels)

with open(f"{OUTPUT_DIR}/cluster_quality.txt", "w", encoding="utf-8") as f:
    f.write("ĐÁNH GIÁ CHẤT LƯỢNG PHÂN CỤM 3 NODE (CLUSTER QUALITY)\n\n")
    f.write(f"Silhouette Score (0–1, càng cao càng phân tách tốt): {sil:.4f}\n")
    f.write(f"Davies-Bouldin Index (càng thấp càng tốt): {db:.4f}\n")
    f.write(f"Calinski–Harabasz (càng cao càng tốt): {ch:.4f}\n")

print(f"   ✔ Xuất cluster_quality.txt")


print("-> [6/7] Ghi báo cáo tổng hợp...")

with open(f"{OUTPUT_DIR}/report_iid.txt", "w", encoding="utf-8") as f:

    f.write("BÁO CÁO PHÂN TÍCH ĐỘ NON-IID DỮ LIỆU FL (FULL SUITE)\n")
    f.write("=" * 90 + "\n\n")

    f.write("[1] Ý NGHĨA CÁC CHỈ SỐ\n")
    f.write(" - Wasserstein Distance (WD): đo độ lệch giá trị (dữ liệu đã scale 0-1)\n")
    f.write("   * < 0.05 : Rất giống (IID)\n")
    f.write("   * 0.05–0.20 : Khác nhẹ\n")
    f.write("   * > 0.20 : Khác đáng kể\n\n")

    f.write(" - Jensen-Shannon Distance (JS): đo khác về hình dạng phân phối\n")
    f.write("   * < 0.10 : Hình dáng tương đồng\n")
    f.write("   * 0.10–0.30 : Khác hình dáng\n")
    f.write("   * > 0.30 : Hình dáng hoàn toàn khác\n\n")

    f.write(" - Hellinger Distance (HD): đo khoảng cách phân phối (0–1)\n")
    f.write("   * < 0.10 : Gần IID\n")
    f.write("   * 0.10–0.25 : Non-IID nhẹ\n")
    f.write("   * 0.25–0.40 : Non-IID rõ rệt\n")
    f.write("   * > 0.40 : Non-IID mạnh\n\n")

    f.write(" - KS-test (p-value): p < 0.05 chỉ ra khác biệt thống kê\n\n")

    f.write("[2] NON-IID SCORE GIỮA CÁC NODE (Tổng hợp các feature)\n")
    for k, v in summary_scores.items():
        f.write(f"   {k:<20}: {v:.4f}\n")
    f.write("\n")

    f.write("Thang đánh giá Non-IID Score (tổng hợp):\n")
    f.write("  < 0.15 : gần IID\n")
    f.write("  0.15–0.35 : Non-IID nhẹ\n")
    f.write("  0.35–0.6  : Non-IID rõ rệt\n")
    f.write("  > 0.6     : Non-IID mạnh\n\n")

    f.write("[3] CHI TIẾT TỪNG FEATURE (WD / JS / HD / KS)\n")
    f.write("-" * 90 + "\n")
    for _, row in metrics_df.iterrows():
        pv = row['KS_Pval']
        pv_str = f"{pv:.1e}" if pv < 0.001 else f"{pv:.4f}"
        f.write(f"{row['Feature']:<30} | WD={row['Wasserstein']:.4f} | JS={row['JS_Dist']:.4f} | "
                f"HD={row['Hellinger']:.4f} | KS={pv_str}\n")

print("\n DONE! Mở thư mục outputs_iid_full/ để xem kết quả." )
