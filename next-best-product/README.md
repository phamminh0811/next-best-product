# Next-Best-Product Recommendation — Du lịch & Taxi/Giao đồ ăn

## Bài toán

Benchmark các họ mô hình gợi ý (matrix factorization, item2vec, sequence model như GRU4Rec/SASRec, LLM-based reranking) cho hai bối cảnh:

- **(a) Du lịch nghỉ dưỡng:** gợi ý điểm đến / hạng phòng / gói kỳ nghỉ, và gợi ý dịch vụ trong kỳ lưu trú (spa, vui chơi, F&B).
- **(b) Di chuyển & giao đồ ăn:** gợi ý quán/món ăn theo ngữ cảnh, dự báo điểm đến chuyến xe, và cross-sell hai chiều taxi ↔ food.

### Câu hỏi nghiên cứu

1. Du lịch có tần suất mua rất thấp, tính mùa vụ và ngữ cảnh chuyến đi (đi một mình / cặp đôi / gia đình) — mô hình session-based/context-aware nào vượt trội so với collaborative filtering truyền thống trên dữ liệu thưa?
2. Với giao đồ ăn, vị trí, khung giờ ăn, thời tiết, thời gian giao dự kiến đóng góp bao nhiêu vào chất lượng gợi ý; cân bằng tái đặt món quen (exploitation) và khám phá quán mới (exploration) thế nào?
3. Cross-sell liên PnL (khách đặt phòng → gợi ý xe đón sân bay; khách hay gọi xe khung giờ trưa → gợi ý food): mô hình hóa ra sao khi hai nguồn dữ liệu tách biệt, không có user_id chung?
4. Đánh đổi giữa độ chính xác (recall@k, NDCG) và độ đa dạng/độ phủ danh mục ở từng bối cảnh?

## Dataset & lý do chọn

| Dataset | Tín hiệu chính | Vai trò trong bài toán |
| --- | --- | --- |



## TODO
- [ ] : EDA
- [ ] : Recommendation for every single product
- [ ] : Cross-sell approach
