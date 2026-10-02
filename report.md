Họ và tên: Lê Tuấn Anh
MSSV: 2A202602952

Kết quả benchmark LightGBM trên `e2-medium` (2 vCPU / 4 GB RAM, x86_64, LightGBM 4.7.0):

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
| Inference latency (1 row) | 1.248 ms (median 50 lần) |
| Inference throughput (1000 rows) | 3.74 ms (~267,469 rows/s, median 10 lần) |

Confusion matrix trên tập test (56,962 giao dịch, 98 ca gian lận):

|  | Đoán: bình thường | Đoán: gian lận |
|---|---|---|
| **Thật: bình thường** | 56,856 (TN) | 8 (FP) |
| **Thật: gian lận** | 20 (FN) | 78 (TP) |

Nhận xét về kết quả training time, AUC, inference speed trên CPU:

- **Training time:** Load ~285k giao dịch (file CSV ~150 MB) mất 2.66 s. Training trên 170,883 dòng (60% dữ liệu) mất 4.10 s với 2 vCPU, early stopping chọn best iteration = 68. Với dữ liệu dạng bảng cỡ này, CPU nhỏ là đủ, không cần GPU.
- **AUC:** AUC-ROC = 0.9768, nghĩa là mô hình xếp hạng ca gian lận cao hơn ca bình thường rất tốt. Ở ngưỡng 0.5, mô hình bắt được 78/98 ca gian lận (Recall 0.80) và chỉ báo nhầm 8 ca (Precision 0.91). Accuracy 0.9995 không nói lên nhiều, vì đoán "bình thường" cho mọi giao dịch cũng đạt 0.9983.
- **Bài học từ lần chạy đầu:** Lần chạy đầu dùng `scale_pos_weight` ≈ 577 cho kết quả hỏng (AUC 0.04, Recall 0). Bỏ tham số này đi thì mô hình học bình thường trở lại. Lần chạy sau cũng đổi thêm cách chia dữ liệu, `n_estimators` và early stopping, nên chưa tách riêng được ảnh hưởng của từng thay đổi.
- **Inference speed:** Dự đoán 1 dòng mất ~1.25 ms. Dự đoán 1000 dòng một lần mất 3.74 ms (~3.7 µs/dòng), nhanh hơn ~330 lần mỗi dòng so với gọi từng dòng, vì chi phí cố định mỗi lần gọi chỉ trả một lần. CPU thừa sức phục vụ cả real-time (từng giao dịch) lẫn batch scoring.
- **Hướng cải thiện:** Vẫn còn bỏ lọt 20 ca gian lận. Nếu ưu tiên bắt gian lận hơn tránh báo nhầm, có thể hạ ngưỡng quyết định xuống dưới 0.5 để tăng Recall, đổi lại Precision sẽ giảm.
