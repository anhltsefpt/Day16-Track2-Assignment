Họ và tên: Lê Tuấn Anh
MSSV: 2A202602952

Kết quả benchmark LightGBM trên `e2-medium` (2 vCPU / 4 GB RAM):

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.683 s |
| Thời gian training | 3.594 s |
| Best iteration | 2 |
| AUC-ROC | 0.0396 |
| Accuracy | 0.9919 |
| F1-Score | 0.0 |
| Precision | 0.0 |
| Recall | 0.0 |
| Inference latency (1 row) | 1.169 ms |
| Inference throughput (1000 rows) | 0.0015 s (~677,637 rows/s) |

Confusion matrix trên tập test (~56,962 giao dịch, ~98 ca gian lận), ước tính ngược từ Accuracy và Recall:

|  | Đoán: bình thường | Đoán: gian lận |
|---|---|---|
| **Thật: bình thường** | ~56,502 (TN) | ~362 (FP) |
| **Thật: gian lận** | 98 (FN) | 0 (TP) |

TP = 0 nên Precision = TP / (TP + FP) = 0, Recall = TP / (TP + FN) = 0, và F1 = 0.

Nhận xét về kết quả training time, AUC, inference speed trên CPU:

- **Training time:** Load ~285k giao dịch (file CSV ~150 MB) mất 2.68 s. Training mất 3.59 s trên `e2-medium` (2 vCPU / 4 GB RAM), early stopping dừng ở vòng 52 (best iteration = 2). Với dữ liệu dạng bảng cỡ này, CPU nhỏ là đủ, không cần GPU.
- **AUC:** AUC-ROC = 0.0396, thấp hơn nhiều so với mức đoán ngẫu nhiên (0.5). Precision, Recall và F1 đều bằng 0, tức mô hình không bắt được ca gian lận nào. Accuracy 0.9919 trông có vẻ cao nhưng thực ra còn thấp hơn cách đoán "bình thường" cho mọi giao dịch (~0.9983), vì gian lận chỉ chiếm 0.17% dữ liệu. Đây là ví dụ cho thấy Accuracy không dùng được để đánh giá dữ liệu mất cân bằng.
- **Nguyên nhân:** AUC trên tập validation đã rất thấp ngay từ vòng 2 (0.0686) rồi tiếp tục giảm, nên training bị phân kỳ từ đầu. Khả năng cao là do `scale_pos_weight` ≈ 577 (tỉ lệ bình thường/gian lận) quá lớn, làm gradient của lớp gian lận lấn át và khiến mô hình học ngược. Hướng khắc phục: bỏ `scale_pos_weight` hoặc giảm xuống (ví dụ ~10), rồi chạy lại.
- **Inference speed:** Dự đoán 1 dòng mất ~1.17 ms. Dự đoán 1000 dòng một lần chỉ mất 1.5 ms (~678k dòng/giây), nhanh hơn ~800 lần mỗi dòng so với gọi từng dòng, nhờ chi phí cố định mỗi lần gọi chỉ trả một lần. Như vậy CPU thừa sức phục vụ cả real-time (từng giao dịch) lẫn batch scoring.
