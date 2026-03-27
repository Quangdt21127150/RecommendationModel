import pandas as pd
import numpy as np
import implicit
from scipy.sparse import coo_matrix
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"

print("Đang load data...")
train_df = pd.read_csv("train_v3.csv")
probe_df = pd.read_csv("probe_v3.csv")

print("Đang xây sparse matrix...")
user_ids = train_df["user_id"].unique()
item_ids = train_df["item_id"].unique()

user_map = {uid: i for i, uid in enumerate(user_ids)}
item_map = {iid: i for i, iid in enumerate(item_ids)}
item_map_rev = {i: iid for iid, i in item_map.items()}

row = train_df["user_id"].map(user_map).values
col = train_df["item_id"].map(item_map).values
data = np.ones(len(train_df), dtype=np.float32)

sparse_matrix = coo_matrix(
    (data, (row, col)), shape=(len(user_ids), len(item_ids))
).tocsr()

print("Đang train ALS model (chỉ trên train_v3.csv)...")

alpha = 40
confidence = sparse_matrix * alpha

model = implicit.als.AlternatingLeastSquares(
    factors=128, regularization=0.05, iterations=80, random_state=42
)

model.fit(confidence)

print("Train ALS xong!")

print("Đang tạo top 50 cho tất cả users...")

all_users = sorted(train_df["user_id"].unique())
submission = {}

popular = train_df["item_id"].value_counts().head(50).index.tolist()

user_items = sparse_matrix

for user in all_users:
    if user not in user_map:
        submission[user] = popular
        continue

    user_idx = user_map[user]

    ids, scores = model.recommend(user_idx, user_items[user_idx], N=100)

    recommended_items = [item_map_rev[i] for i in ids if i in item_map_rev]

    submission[user] = recommended_items[:50]

print("Đang ghi file submission...")
with open("submission_als.txt", "w") as f:
    for user in all_users:
        line = " ".join(map(str, submission[user]))
        f.write(line + "\n")

print("✅ ĐÃ TẠO XONG FILE: submission_als.txt (sẵn sàng nộp)")

print("Đang tính NDCG@50, NCRR@50, Recall@50 và Harmonic Mean...")


def calculate_metrics(probe_df, submission):
    ndcg_list, ncrr_list, recall_list = [], [], []

    for user_id, group in probe_df.groupby("user_id"):
        ground_truth = dict(zip(group["item_id"], group["rating"]))
        if not ground_truth:
            continue

        predicted = submission.get(user_id, [])[:50]

        hits = sum(1 for item in predicted if item in ground_truth)
        recall = hits / len(ground_truth)
        recall_list.append(recall)

        dcg = sum(
            ground_truth.get(item, 0) / math.log2(i + 2)
            for i, item in enumerate(predicted)
        )
        ideal = sorted(ground_truth.values(), reverse=True)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_list.append(ndcg)

        crr = sum(
            1.0 / (i + 1) for i, item in enumerate(predicted) if item in ground_truth
        )
        ideal_crr = sum(1.0 / (j + 1) for j in range(len(ground_truth)))
        ncrr = crr / ideal_crr if ideal_crr > 0 else 0.0
        ncrr_list.append(ncrr)

    avg_ndcg = np.mean(ndcg_list)
    avg_ncrr = np.mean(ncrr_list)
    avg_recall = np.mean(recall_list)
    harmonic = 3 / (1 / avg_ndcg + 1 / avg_ncrr + 1 / avg_recall + 1e-8)

    return avg_ndcg, avg_ncrr, avg_recall, harmonic


ndcg, ncrr, recall, harmonic = calculate_metrics(probe_df, submission)

print("\n=== KẾT QUẢ MODEL ALS ===")
print(f"NDCG@50     : {ndcg:.5f}")
print(f"NCRR@50     : {ncrr:.5f}")
print(f"Recall@50   : {recall:.5f}")
print(f"**Harmonic Mean : {harmonic:.5f}**")
