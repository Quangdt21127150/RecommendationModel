import os
import math
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import hmean
import implicit
import lightgbm as lgb
from collections import defaultdict, Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

print("Đang load data...")
train_df = pd.read_csv("train_v3.csv")
probe_df = pd.read_csv("probe_v3.csv")

# ==================== MAPPING ====================
user_ids = train_df["user_id"].unique()
item_ids = train_df["item_id"].unique()
user_map = {uid: i for i, uid in enumerate(user_ids)}
item_map = {iid: i for i, iid in enumerate(item_ids)}
item_map_rev = {i: iid for iid, i in item_map.items()}

# Sparse matrix
row = train_df["user_id"].map(user_map).values
col = train_df["item_id"].map(item_map).values
sparse_matrix = coo_matrix(
    (np.ones(len(train_df), dtype=np.float32), (row, col)),
    shape=(len(user_ids), len(item_ids)),
).tocsr()

# ==================== ALS ====================
print("Đang train ALS...")
model = implicit.als.AlternatingLeastSquares(
    factors=128, regularization=0.05, iterations=30, random_state=42
)
model.fit(sparse_matrix * 40)

# ==================== CANDIDATES ====================
print("Đang tạo candidates...")

user_history = train_df.groupby("user_id")["item_id"].apply(list).to_dict()
co_counts = defaultdict(Counter)

for items in user_history.values():
    uniq = list(set(items))
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            co_counts[a][b] += 1
            co_counts[b][a] += 1

cand_users, cand_items, cand_scores, cand_rank, cand_co = [], [], [], [], []

for user in user_ids:
    hist = user_history.get(user, [])
    hist_set = set(hist)
    u_idx = user_map[user]

    # ALS
    ids, scores = model.recommend(
        u_idx, sparse_matrix[u_idx], N=800, filter_already_liked_items=True
    )
    for r, (iid, s) in enumerate(zip(ids, scores), 1):
        cand_users.append(user)
        cand_items.append(item_map_rev[iid])
        cand_scores.append(s)
        cand_rank.append(r)
        cand_co.append(0)

    # Co-occurrence
    for item in hist[:10]:
        for sim_item, c in co_counts[item].most_common(100):
            if sim_item not in hist_set:
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

# ==================== FEATURES ====================
print("Đang tạo features...")

cand_df = cand_df.join(train_df["item_id"].value_counts().rename("pop"), on="item_id")
cand_df = cand_df.join(
    train_df.groupby("item_id")["rating"].mean().rename("item_mean"), on="item_id"
)
cand_df = cand_df.join(
    train_df.groupby("item_id").size().rename("item_cnt"), on="item_id"
)
cand_df = cand_df.join(
    train_df.groupby("user_id")["rating"].mean().rename("user_mean"), on="user_id"
)
cand_df = cand_df.join(
    train_df.groupby("user_id").size().rename("user_cnt"), on="user_id"
)

cand_df["log_pop"] = np.log1p(cand_df["pop"])
cand_df["log_co"] = np.log1p(cand_df["co"])

# Ground truth
gt = probe_df[["user_id", "item_id", "rating"]]
cand_df = cand_df.merge(gt, on=["user_id", "item_id"], how="left")
cand_df["rating"] = cand_df["rating"].fillna(0)

# ==================== LIGHTGBM RANKER ====================
print("Đang train LightGBM Ranker...")

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
groups = cand_df.groupby("user_id").size().values

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

# ==================== SUBMISSION ====================
print("Đang tạo submission...")

cand_df["score"] = ranker.predict(X)

submission = {
    user: df.sort_values("score", ascending=False)["item_id"].head(50).tolist()
    for user, df in cand_df.groupby("user_id")
}

# Fallback
popular = train_df["item_id"].value_counts().head(50).index.tolist()
for user in probe_df["user_id"].unique():
    if user not in submission:
        submission[user] = popular

# Save
with open("submission_als.txt", "w") as f:
    for user in sorted(submission.keys()):
        f.write(" ".join(map(str, submission[user])) + "\n")

print("✅ ĐÃ TẠO XONG FILE: submission_als.txt")


# ==================== METRICS ====================
def calculate_metrics(probe_df, submission):
    ndcg_list, ncrr_list, recall_list = [], [], []

    for user_id, group in probe_df.groupby("user_id"):
        gt = dict(zip(group["item_id"], group["rating"]))
        if not gt:
            continue

        pred = submission.get(user_id, [])[:50]
        hits = sum(1 for i in pred if i in gt)

        # NDCG
        dcg = sum(gt.get(i, 0) / math.log2(k + 2) for k, i in enumerate(pred))
        idcg = sum(
            g / math.log2(k + 2)
            for k, g in enumerate(sorted(gt.values(), reverse=True))
        )
        ndcg_list.append(dcg / idcg if idcg > 0 else 0)

        # NCRR
        crr = sum(1.0 / (k + 1) for k, i in enumerate(pred) if i in gt)
        ideal_crr = sum(1.0 / (k + 1) for k in range(len(gt)))
        ncrr_list.append(crr / ideal_crr if ideal_crr > 0 else 0)

        recall_list.append(hits / len(gt))

    avg_ndcg = np.mean(ndcg_list)
    avg_ncrr = np.mean(ncrr_list)
    avg_recall = np.mean(recall_list)
    harmonic = hmean([avg_ndcg, avg_ncrr, avg_recall])

    return avg_ndcg, avg_ncrr, avg_recall, harmonic


ndcg, ncrr, recall, harmonic = calculate_metrics(probe_df, submission)

print("\n=== KẾT QUẢ MODEL ALS ===")
print(f"NDCG@50     : {ndcg:.5f}")
print(f"NCRR@50     : {ncrr:.5f}")
print(f"Recall@50   : {recall:.5f}")
print(f"**Harmonic Mean : {harmonic:.5f}**")
