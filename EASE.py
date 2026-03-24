import cornac
from cornac.metrics import Recall, NDCG, NCRR
import numpy as np
import pandas as pd
import os
import statistics

SEED = 42
TOPK = 50
LAMB = 3203
POSB = True
VERBOSE = True

print("Đang load dữ liệu...")

df_train = pd.read_csv(
    "train_v3.csv", dtype={"user_id": str, "item_id": str, "rating": float}
)
train_triples = list(
    df_train[["user_id", "item_id", "rating"]].itertuples(index=False, name=None)
)

df_probe = pd.read_csv(
    "probe_v3.csv", dtype={"user_id": str, "item_id": str, "rating": float}
)
probe_triples = list(
    df_probe[["user_id", "item_id", "rating"]].itertuples(index=False, name=None)
)

print(f"Số interaction train: {len(train_triples)}")
print(f"Số interaction probe: {len(probe_triples)}")

train_set = cornac.data.Dataset.from_uir(train_triples, seed=SEED)

print("\nTrain model EASE trên train set...")

model = cornac.models.EASE(
    lamb=LAMB,
    posB=POSB,
    seed=SEED,
    verbose=VERBOSE,
)

model.fit(train_set)

print("\nĐánh giá trên probe...")

probe_set = cornac.data.Dataset.build(
    data=probe_triples,
    fmt="UIR",
    global_uid_map=train_set.uid_map,
    global_iid_map=train_set.iid_map,
    seed=SEED,
)

ranking_results, _ = cornac.eval_methods.ranking_eval(
    model=model,
    train_set=train_set,
    test_set=probe_set,
    metrics=[Recall(k=TOPK), NDCG(k=TOPK), NCRR(k=TOPK)],
    verbose=VERBOSE,
)

print("\n" + "=" * 80)
print("KẾT QUẢ TRÊN PROBE (EASE):")
print("-" * 60)
recall, ndcg, ncrr = ranking_results

print(f"Recall@50 : {recall:.6f}")
print(f"NDCG@50   : {ndcg:.6f}")
print(f"NCRR@50   : {ncrr:.6f}")
print(f"Harmonic Mean   : {statistics.harmonic_mean([recall, ndcg, ncrr]):.6f}")
print("=" * 80)

print("\nTạo submission.txt...")

train_matrix = model.train_set.matrix
B = model.B
scores = train_matrix.dot(B)
uid_map = model.train_set.uid_map
iid_reverse_map = {v: k for k, v in model.train_set.iid_map.items()}
submission = []

for user in range(46612):
    uid = str(user + 1)

    if uid not in uid_map:
        submission.append(" ".join(["1"] * 50))
        continue

    uid_idx = uid_map[uid]

    s = scores[uid_idx]
    if hasattr(s, "toarray"):
        s = s.toarray().ravel()

    seen = train_matrix[uid_idx].indices
    s[seen] = -np.inf

    top_items = np.argpartition(-s, TOPK)[:TOPK]
    top_items = top_items[np.argsort(-s[top_items])]

    top_original = [str(iid_reverse_map[i]) for i in top_items]
    submission.append(" ".join(top_original))

with open("submission.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(submission))

print(f"Hoàn tất! File submission.txt tại: {os.path.abspath('submission.txt')}")
print(f"- Số dòng: {len(submission)}")
