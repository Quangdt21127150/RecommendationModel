import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import implicit
from scipy.sparse import coo_matrix
import math
import lightgbm as lgb
from collections import defaultdict, Counter

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
    factors=128, regularization=0.05, iterations=30, random_state=42
)

model.fit(confidence)

print("Train ALS xong!")

print("Đang tạo candidate items...")

user_items = sparse_matrix

cand_users = []
cand_items = []
cand_scores = []
cand_rank = []
cand_co = []

user_history = train_df.groupby("user_id")["item_id"].apply(list).to_dict()

co_counts = defaultdict(Counter)

for items in user_history.values():
    uniq = list(set(items))
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a = uniq[i]
            b = uniq[j]
            co_counts[a][b] += 1
            co_counts[b][a] += 1

for user in user_ids:

    hist = user_history.get(user, [])
    hist_set = set(hist)

    user_idx = user_map[user]

    ids, scores = model.recommend(
        user_idx, user_items[user_idx], N=800, filter_already_liked_items=True
    )

    for r, (i, s) in enumerate(zip(ids, scores), 1):
        if i in item_map_rev:
            cand_users.append(user)
            cand_items.append(item_map_rev[i])
            cand_scores.append(s)
            cand_rank.append(r)
            cand_co.append(0)

    for item in hist[:10]:
        if item not in co_counts:
            continue
        for sim_item, c in co_counts[item].most_common(100):
            if sim_item in hist_set:
                continue
            cand_users.append(user)
            cand_items.append(sim_item)
            cand_scores.append(0)
            cand_rank.append(999)
            cand_co.append(c)

cand_df = pd.DataFrame(
    {
        "user_id": cand_users,
        "item_id": cand_items,
        "als": cand_scores,
        "rank": cand_rank,
        "co": cand_co,
    }
)

print("Đang tạo features cho ranking...")

popularity = train_df["item_id"].value_counts().rename("pop")
item_mean = train_df.groupby("item_id")["rating"].mean().rename("item_mean")
user_mean = train_df.groupby("user_id")["rating"].mean().rename("user_mean")
user_cnt = train_df.groupby("user_id").size().rename("user_cnt")
item_cnt = train_df.groupby("item_id").size().rename("item_cnt")

cand_df = cand_df.join(popularity, on="item_id")
cand_df = cand_df.join(item_mean, on="item_id")
cand_df = cand_df.join(item_cnt, on="item_id")
cand_df = cand_df.join(user_mean, on="user_id")
cand_df = cand_df.join(user_cnt, on="user_id")

cand_df["log_pop"] = np.log1p(cand_df["pop"])
cand_df["log_co"] = np.log1p(cand_df["co"])

gt = probe_df[["user_id", "item_id", "rating"]]
cand_df = cand_df.merge(gt, on=["user_id", "item_id"], how="left")
cand_df["rating"] = cand_df["rating"].fillna(0)

groups = cand_df.groupby("user_id").size().values

X = cand_df[
    [
        "als",
        "rank",
        "log_pop",
        "log_co",
        "item_mean",
        "user_mean",
        "user_cnt",
        "item_cnt",
    ]
]

y = cand_df["rating"].values

print("Đang train ranking model...")

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=63,
    force_row_wise=True,
    n_jobs=4,
)

ranker.fit(X, y, group=groups)

print("Đang tạo top 50 cho tất cả users...")

cand_df["score"] = ranker.predict(X)

submission = {}

for user, df in cand_df.groupby("user_id"):
    submission[user] = (
        df.sort_values("score", ascending=False)["item_id"].head(50).tolist()
    )

popular = train_df["item_id"].value_counts().head(50).index.tolist()

for user in probe_df["user_id"].unique():
    if user not in submission:
        submission[user] = popular

print("Đang ghi file submission...")
with open("submission_als.txt", "w") as f:
    for user in sorted(submission.keys()):
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
