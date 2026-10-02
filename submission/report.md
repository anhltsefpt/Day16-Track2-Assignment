Họ và tên: Lê Tuấn Anh
MSSV: 2A202602952

1. Tôi dùng GCP, region `us-central1` / zone `us-central1-a`, instance `e2-medium` (2 vCPU / 4 GB RAM, x86_64), source commit `55539f6` (thư mục `terraform-gcp/`, chép trong `infra/`).
2. Dataset Credit Card Fraud có 284,807 dòng (492 ca gian lận, 0.17%), chia train/validation/test = 60/20/20 (170,883 / 56,962 / 56,962 dòng) có stratify theo `Class`, seed 16.
3. Load dữ liệu mất 2.66 giây; training mất 4.10 giây (LightGBM 4.7.0, `n_jobs=2`, early stopping 20 vòng); best iteration là 68.
4. AUC 0.9768, Accuracy 0.9995, F1 0.8478, Precision 0.9070, Recall 0.7959 trên tập test (ngưỡng 0.5: bắt được 78/98 ca gian lận, báo nhầm 8 ca).
5. Latency 1 dòng 1.25 ms; throughput batch 1.000 dòng ~267,469 dòng/giây (3.74 ms/batch); cách đo: `predict_proba` trên pandas, warm-up trước, lấy median của 50 lần (1 dòng) và 10 lần (batch).
6. CPU/RAM/Network tôi quan sát lúc 16:47 (UTC+7), sau lần chạy benchmark đầu và VM đã nhàn rỗi, là CPU idle 99.8%, RAM dùng 488 MiB / 3.8 GiB, `ens4` nhận ~257 MB (tải dataset + package) và gửi ~1.1 MB; ảnh đính kèm `screenshots/top-h.png`, `free-h.png`, `ip-s-link.png`.
7. Billing Reports tại 17:22 ngày 02/10/2026 (UTC+7) ghi nhận ₫0 và "No results to display" (GCP chưa cập nhật chi phí); ước tính riêng ~1 giờ chạy × ~$0.09/giờ (Compute Engine `e2-medium` + Cloud NAT + Load Balancer) ≈ $0.09, được trừ vào free trial credit; ảnh `screenshots/billing.png`.
8. Tôi đã tải kết quả về lúc 17:13 và chạy `terraform destroy` lúc 17:20, hoàn tất lúc ~17:28 (UTC+7); bằng chứng dọn dẹp: `terraform state list` trả về 0 tài nguyên, và `gcloud compute instances list`, `gcloud compute routers list`, `gcloud compute forwarding-rules list` đều trả về "Listed 0 items.".

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.663 s |
| Thời gian training | 4.104 s |
| Best iteration | 68 |
| AUC-ROC | 0.9768 |
| Accuracy | 0.9995 |
| F1-Score | 0.8478 |
| Precision | 0.9070 |
| Recall | 0.7959 |
| Inference latency (1 row) | 1.248 ms |
| Inference throughput (1000 rows) | 3.74 ms (~267,469 rows/s) |

Ghi chú: lần chạy đầu dùng `scale_pos_weight` ≈ 577 cho kết quả hỏng (AUC 0.04, Recall 0); bỏ tham số này thì mô hình học bình thường. Lần chạy lại đồng thời đổi cách chia dữ liệu, `n_estimators` và early stopping, nên chưa tách riêng được ảnh hưởng của từng thay đổi.
