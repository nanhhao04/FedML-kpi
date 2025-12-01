import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import warnings

'''
PHÂN TÍCH CÁC ĐỘ ĐO ĐÁNH GIÁ DỮ LIỆU IID
'''

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs_iid_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"   TOOL ĐÁNH GIÁ ĐỘ NON-IID DỮ LIỆU (FULL SUITE)")
print("=" * 60)
print(f"-> Kết quả sẽ được lưu tại thư mục: {OUTPUT_DIR}/")



def load_data(client_id):
    path = f"data/data_hl19_node_{client_id}.csv"
    try:
        df = pd.read_csv(path)
        # 1.1 Loại bỏ các cột không phải Feature (Timestamp, ID, Label)
        drop_cols = ['date', 'NODE_ID', 'timestamp', 'label', 'anomaly']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        # Chỉ giữ lại cột số
        df = df.select_dtypes(include=[np.number])
        return df
    except FileNotFoundError:
        return None


print("-> [1/5] Đang load dữ liệu 3 Node...")
df1 = load_data(1)
df2 = load_data(2)
df3 = load_data(3)

if df1 is None or df2 is None or df3 is None:
    print("ERROR: Không tìm thấy đủ file dữ liệu data/data_hl19_node_X.csv")
    exit()

# Lấy danh sách feature chung
features = df1.columns.tolist()
print(f"   Tìm thấy {len(features)} features dạng số.")

# scaling
print("-> [2/5] Thực hiện Global Scaling (MinMax 0-1)...")
# Lý do: Để so sánh Wasserstein Distance công bằng giữa các cột có đơn vị khác nhau.

# Gộp data để tìm min/max chung
combined_raw = pd.concat([df1, df2, df3], axis=0)
scaler = MinMaxScaler()
scaler.fit(combined_raw)

# Transform từng node (giữ nguyên sự khác biệt phân phối, chỉ đưa về cùng hệ quy chiếu)
df1_scaled = pd.DataFrame(scaler.transform(df1), columns=features)
df2_scaled = pd.DataFrame(scaler.transform(df2), columns=features)
df3_scaled = pd.DataFrame(scaler.transform(df3), columns=features)

print("-> [3/5] Đang tính toán Wasserstein, KS-Test, JS Distance...")


def calculate_js_distance(p, q, bins=50):
    """Tính Jensen-Shannon Distance trên dữ liệu histogram"""
    range_min = min(p.min(), q.min())
    range_max = max(p.max(), q.max())
    # Tạo histogram xác suất
    p_hist, _ = np.histogram(p, bins=bins, range=(range_min, range_max), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(range_min, range_max), density=True)
    # Tránh lỗi chia cho 0
    p_hist = np.where(p_hist == 0, 1e-10, p_hist)
    q_hist = np.where(q_hist == 0, 1e-10, q_hist)
    return jensenshannon(p_hist, q_hist)


results = []

for col in features:
    # Dùng dữ liệu đã scale để tính Wasserstein
    d1, d2, d3 = df1_scaled[col], df2_scaled[col], df3_scaled[col]

    # A. Wasserstein (Đã scale về 0-1 nên giá trị này so sánh được)
    ws_12 = wasserstein_distance(d1, d2)
    ws_13 = wasserstein_distance(d1, d3)
    ws_23 = wasserstein_distance(d2, d3)
    avg_ws = np.mean([ws_12, ws_13, ws_23])

    # B. KS-Test (P-value)
    # P-value < 0.05 => Khác biệt. Lấy min để bắt lỗi khắt khe nhất.
    ks_12 = ks_2samp(d1, d2).pvalue
    ks_13 = ks_2samp(d1, d3).pvalue
    ks_23 = ks_2samp(d2, d3).pvalue
    min_pval = min(ks_12, ks_13, ks_23)

    # C. Jensen-Shannon Distance (Hình dáng phân phối)
    js_12 = calculate_js_distance(d1, d2)
    js_13 = calculate_js_distance(d1, d3)
    js_23 = calculate_js_distance(d2, d3)
    avg_js = np.mean([js_12, js_13, js_23])

    results.append({
        "Feature": col,
        "Wasserstein": avg_ws,
        "KS_Pval": min_pval,
        "JS_Dist": avg_js
    })

metrics_df = pd.DataFrame(results)


print("-> [4/5] Vẽ biểu đồ PCA và KDE...")

# A. PCA PLOT (Không gian đặc trưng)

n_samp = 500
s1 = df1_scaled.sample(min(n_samp, len(df1)))
s2 = df2_scaled.sample(min(n_samp, len(df2)))
s3 = df3_scaled.sample(min(n_samp, len(df3)))

X_pca_input = np.vstack([s1, s2, s3])
y_labels = ['Node 1'] * len(s1) + ['Node 2'] * len(s2) + ['Node 3'] * len(s3)

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_pca_input)  # Data đã scale min-max rồi

plt.figure(figsize=(9, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, style=y_labels, alpha=0.7, s=60)
plt.title("Không gian dữ liệu 3 Node (PCA trên dữ liệu đã Scale)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/1_pca_structure.png")
plt.close()

# Chọn 3 feature có Wasserstein cao nhất
top_diff = metrics_df.sort_values("Wasserstein", ascending=False).head(3)["Feature"].tolist()

plt.figure(figsize=(18, 5))
for i, col in enumerate(top_diff):
    plt.subplot(1, 3, i + 1)
    # Vẽ trên dữ liệu gốc (chưa scale) để nhìn giá trị thực tế cho dễ hiểu
    sns.kdeplot(df1[col], label='Node 1', fill=True, alpha=0.3)
    sns.kdeplot(df2[col], label='Node 2', fill=True, alpha=0.3)
    sns.kdeplot(df3[col], label='Node 3', fill=True, alpha=0.3)
    plt.title(f"Non-IID Feature: {col}")
    plt.legend()
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_kde_top_diff.png")
plt.close()

# report
print("-> [5/5] Đang ghi báo cáo phân tích...")
report_path = f"{OUTPUT_DIR}/report_iid.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("          BÁO CÁO PHÂN TÍCH ĐỘ NON-IID DỮ LIỆU FL\n")

    f.write("[1] HƯỚNG DẪN ĐỌC CHỈ SỐ (QUAN TRỌNG)\n")
    f.write("-" * 60 + "\n")
    f.write("Dữ liệu đã được scale về [0, 1] trước khi tính toán.\n\n")

    f.write("1. Wasserstein Distance (WD) [0.0 - 1.0]:\n")
    f.write("   - Đo khoảng cách giữa 2 phân phối.\n")
    f.write("   - < 0.05 : Rất giống nhau (IID).\n")
    f.write("   - 0.05 - 0.2 : Khác biệt nhẹ (Mild Non-IID).\n")
    f.write("   - > 0.2 : Khác biệt lớn (Strong Non-IID - Lệch về độ lớn).\n\n")

    f.write("2. Jensen-Shannon Distance (JSD) [0.0 - 1.0]:\n")
    f.write("   - Đo sự khác biệt về HÌNH DÁNG biểu đồ phân phối.\n")
    f.write("   - < 0.1 : Hình dáng tương đồng.\n")
    f.write("   - > 0.3 : Hình dáng hoàn toàn khác nhau.\n\n")

    f.write("3. KS-Test P-value:\n")
    f.write("   - < 0.05 : Khẳng định thống kê là 2 Node KHÁC NHAU.\n")
    f.write("   - Lưu ý: Với dữ liệu lớn, p-value luôn rất nhỏ.\n\n")

    # --- PHẦN 2: BẢNG KẾT QUẢ ---
    f.write("[2] CHI TIẾT TỪNG FEATURE\n")
    f.write("-" * 95 + "\n")
    f.write(f"{'Feature':<25} | {'Wasserstein':<12} | {'JS Dist':<12} | {'KS P-val':<12} | {'Đánh giá'}\n")
    f.write("-" * 95 + "\n")

    for _, row in metrics_df.iterrows():
        wd = row['Wasserstein']
        js = row['JS_Dist']
        pv = row['KS_Pval']

        # Logic đánh giá tự động
        if wd < 0.05 and js < 0.1:
            rating = "IID (Tốt)"
        elif wd > 0.2:
            rating = "Non-IID (Lệch Giá trị)"
        elif js > 0.3:
            rating = "Non-IID (Lệch Hình dáng)"
        else:
            rating = "Non-IID (Nhẹ)"

        p_str = f"{pv:.1e}" if pv < 0.001 else f"{pv:.3f}"
        f.write(f"{row['Feature']:<25} | {wd:<12.4f} | {js:<12.4f} | {p_str:<12} | {rating}\n")

    f.write("-" * 95 + "\n\n")
    f.write("GHI CHÚ: Xem hình ảnh '1_pca_structure.png' và '2_kde_top_diff.png' để trực quan hóa.")

print(f"\n[DONE] Hoàn tất! Hãy mở file báo cáo tại: {report_path}")