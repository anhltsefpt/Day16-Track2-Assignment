import json, time
import pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

load_start = time.time()
# Lấy creditcard.csv từ https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
transactions = pd.read_csv("creditcard.csv")
load_time_s = time.time() - load_start

features, labels = transactions.drop(columns=["Class"]), transactions["Class"]
features_trainval, features_test, labels_trainval, labels_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42)
features_train, features_val, labels_train, labels_val = train_test_split(
    features_trainval, labels_trainval, test_size=0.1, stratify=labels_trainval, random_state=42)

num_normal = (labels_train == 0).sum()
num_fraud = (labels_train == 1).sum()
model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           scale_pos_weight=num_normal / num_fraud, verbose=-1)
train_start = time.time()
model.fit(features_train, labels_train, eval_set=[(features_val, labels_val)], eval_metric="auc",
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
train_time_s = time.time() - train_start

fraud_probability = model.predict_proba(features_test)[:, 1]
predicted_labels = (fraud_probability >= 0.5).astype(int)

single_row = features_test.iloc[[0]]
num_repeats = 100
latency_start = time.perf_counter()
for _ in range(num_repeats):
    model.predict_proba(single_row)
latency_per_row_ms = (time.perf_counter() - latency_start) / num_repeats * 1000

batch_size = 1000
batch_rows = features_test.iloc[:batch_size]
batch_start = time.perf_counter()
model.predict_proba(batch_rows)
batch_time_s = time.perf_counter() - batch_start

result = {
    "load_time_s": round(load_time_s, 3),
    "train_time_s": round(train_time_s, 3),
    "best_iteration": int(model.best_iteration_),
    "auc_roc": round(roc_auc_score(labels_test, fraud_probability), 4),
    "accuracy": round(accuracy_score(labels_test, predicted_labels), 4),
    "f1": round(f1_score(labels_test, predicted_labels), 4),
    "precision": round(precision_score(labels_test, predicted_labels), 4),
    "recall": round(recall_score(labels_test, predicted_labels), 4),
    "latency_1row_ms": round(latency_per_row_ms, 3),
    "throughput_1000rows_s": round(batch_time_s, 4),
    "throughput_rows_per_s": round(batch_size / batch_time_s, 1),
}
print(json.dumps(result, indent=2))
with open("benchmark_result.json", "w") as output_file:
    json.dump(result, output_file, indent=2)
