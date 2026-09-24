# Methodology cho hệ thống gợi ý đặt phòng khách sạn

## 1. Bài toán next-best-product cho đặt phòng khách sạn

Domain du lịch – khách sạn có những đặc điểm sau ảnh hưởng trực tiếp đến việc chọn phương pháp luận cho bài toán:

- Tần suất mua rất thấp – khách thường chỉ đặt phòng một đến hai lần mỗi năm, khiến tín hiệu tương tác dài hạn giữa cùng một người dùng và khách sạn rất thưa.
- Ngữ cảnh chuyến đi quyết định lựa chọn – cùng một người dùng nhưng đi một mình, đi cặp đôi, đi cùng gia đình hay nhóm lớn sẽ có nhu cầu khác nhau giữa các lần đặt phòng.
- Cold-start xảy ra ở cả hai chiều – vừa có người dùng mới với rất ít lượt đặt phòng, vừa có khách sạn mới chưa tích lũy đủ lịch sử tương tác.

Như đã trình bày trước đó, bộ dữ liệu được chọn là Expedia Hotel Recommendations. Sau quá trình EDA, ta đã phân tích được bộ dữ liệu có những đặc điểm sau:

- Quy mô lớn nhưng feedback tích cực rất hiếm, ma trận tương tác giữa user và khách sạn cực kỳ thưa.
- Phần lớn người dùng là cold-start.
- Có sẵn ma trận user × hotel, đồng thời có thể tách biệt được lượt xem (click) và lượt đặt (book) theo từng user.
- Không có đặc tính cụ thể của từng khách sạn như hạng sao, tiện nghi, hay giá phòng.

→ Những đặc điểm trên cho thấy Collaborative Filtering là phù hợp với bộ dữ liệu này.

## 2. Các phương pháp hiện đại giải quyết bài toán

Sau khi tìm hiểu và phân tích các phương pháp SOTA hiện nay, các kỹ thuật Collaborative Filtering được chia thành hai hướng phát triển chính:

- **Graph-based**: biến đổi lịch sử tương tác của user thành một đồ thị (graph), rồi lan truyền thông tin qua nhiều bước để suy luận sở thích gián tiếp.
- **Session-based**: ngược lại, chỉ nhìn vào chuỗi hành vi trong một phiên truy cập hiện tại, không cần biết lịch sử dài hạn hay danh tính cố định của user.

Nhiều kỹ thuật huấn luyện quan trọng được sử dụng ở cả hai hướng tiếp cận nêu trên, chỉ khác nhau ở việc áp dụng lên loại dữ liệu nào. Ba kỹ thuật cốt lõi được trình bày dưới đây.

### 2.1. Negative Sampling (lấy mẫu âm)

Khi huấn luyện mô hình, cần dạy mô hình phân biệt giữa "sản phẩm người dùng thực sự quan tâm" (positive sample) và "sản phẩm người dùng không quan tâm" (negative sample). Trong thực tế, số lượng khách sạn mà một người dùng không tương tác luôn lớn hơn rất nhiều lần so với số khách sạn họ có tương tác; nếu đưa toàn bộ số khách sạn còn lại vào mỗi lần huấn luyện thì sẽ quá tốn thời gian và tài nguyên tính toán.

Negative sampling là kỹ thuật chọn ra một số lượng nhỏ khách sạn trong nhóm "chưa tương tác" để làm ví dụ minh hoạ cho việc "không quan tâm", thay vì sử dụng toàn bộ danh mục. Cách chọn mẫu âm càng có chọn lọc – ví dụ ưu tiên chọn những khách sạn dễ gây nhầm lẫn với khách sạn thực sự được quan tâm, gọi là hard negative – thì mô hình học được càng chính xác.

### 2.2. Contrastive Learning (học đối chiếu)

Đây là cách huấn luyện mô hình mà không cần dữ liệu được gán nhãn sẵn. Ý tưởng cốt lõi: tạo ra hai phiên bản hơi khác nhau của cùng một dữ liệu gốc – ví dụ hai phiên bản hơi khác nhau của cùng một graph, hoặc hai session được biến đổi nhẹ – rồi huấn luyện mô hình nhận ra rằng hai phiên bản này vẫn đại diện cho "cùng một đối tượng", đồng thời phân biệt rõ với các đối tượng khác trong tập dữ liệu.

Kỹ thuật này đặc biệt hữu ích khi dữ liệu tương tác thực tế có sẵn rất ít hoặc thưa thớt, như trong trường hợp đặt khách sạn, vì nó giúp mô hình học ra một biểu diễn ổn định cho từng người dùng và khách sạn, thay vì "học thuộc" theo từng chi tiết nhỏ lẻ của dữ liệu quan sát được, vốn dễ chứa nhiễu.

### 2.3. Sampled Softmax (hàm mất mát dạng softmax có lấy mẫu)

Đây là một dạng hàm mất mát (loss function – thước đo sai số dùng để huấn luyện mô hình) kết hợp cả hai ý tưởng nêu trên. Thay vì so sánh điểm số của khách sạn đúng với toàn bộ danh mục khách sạn – điều rất tốn kém khi danh mục có hàng chục nghìn lựa chọn – Sampled Softmax chỉ so sánh với một tập nhỏ mẫu âm được lấy mẫu theo kỹ thuật negative sampling, dựa trên công thức toán của hàm softmax nhằm xác định lựa chọn nào có khả năng đúng cao nhất. Đây là kỹ thuật được nhiều nghiên cứu mới nhất trong giai đoạn 2025–2026 sử dụng, vừa đảm bảo độ chính xác cao, vừa giữ tốc độ huấn luyện hợp lý trên danh mục sản phẩm lớn.

## 3. Nghiên cứu sâu 5 thuật toán cụ thể

Dù cả hai hướng đều khả thi với bộ dữ liệu Expedia, dự án quyết định phát triển sâu theo hướng Graph-based trước vì:

- Định nghĩa session trên dữ liệu Expedia chưa thực sự rõ ràng: kết quả EDA cho thấy các cách định nghĩa session khác nhau (theo ngày, theo khoảng nghỉ 30 phút, theo destination) cho ra tỷ lệ chốt rất khác nhau, từ 7,97% đến 25,78%, nghĩa là cần thêm công sức tiền xử lý và thử nghiệm để chọn đúng định nghĩa trước khi có thể huấn luyện một mô hình Session-based đáng tin cậy.
- Ma trận user × hotel_cluster để làm đã có sẵn và rõ ràng ngay từ dữ liệu gốc, không cần thêm bước xử lý trung gian, giúp rút ngắn thời gian từ nghiên cứu đến có baseline chạy được.

Trong hướng Graph-based Collaborative Filtering, các nghiên cứu qua từng giai đoạn đã đi theo một quá trình phát triển khá rõ rệt. Giai đoạn đầu tập trung đơn giản hoá kiến trúc mạng nơ-ron để việc huấn luyện trở nên dễ dàng, nhanh và ổn định hơn. Giai đoạn sau, tập trung làm giàu chất lượng huấn luyện bằng các kỹ thuật self-supervised learning và contrastive learning, đặc biệt là các phương pháp denoising graph – tức lọc bớt những mối liên kết không đáng tin cậy hoặc bù đắp những mối liên kết còn thiếu trong dữ liệu quan sát được, trước khi đưa vào mô hình.

Một điểm đáng chú ý là các nghiên cứu càng về sau càng cho thấy phần "kiến trúc lan truyền", tức cách mạng nơ-ron xử lý đồ thị, hầu như không còn thay đổi nhiều so với giai đoạn đầu – phần lớn vẫn dựa trên công thức lan truyền tuyến tính đơn giản đã được chứng minh hiệu quả từ sớm. Sự cải tiến trong các nghiên cứu mới chủ yếu đến từ hai chỗ khác: cách xây dựng và làm sạch đồ thị đầu vào, và cách thiết kế hàm mất mát, đặc biệt là cách lấy mẫu âm và áp dụng contrastive learning.

Để kiểm chứng lựa chọn trên bằng bằng chứng cụ thể thay vì chỉ lý luận chung, nhóm nghiên cứu sâu 5 thuật toán: **LightGCL** (ICLR 2023) là mô hình được chọn làm baseline thực tế, cộng với 4 thuật toán khác dùng để so sánh và định hướng lộ trình mở rộng – 2 thuật toán Graph-based (**SCCF** – KDD 2024; **RaDAR** – WWW 2026, mới nhất) và 2 thuật toán Session-based (**SASRec/TiSASRec** – baseline sequential kinh điển, được dùng rất phổ biến; **TRON** – RecSys 2023, mô hình production của OTTO). Với mỗi thuật toán, nhóm tìm hiểu 6 khía cạnh: ý tưởng, nền tảng toán học cơ bản, điểm mạnh/điểm yếu, giới hạn và tính ứng dụng thực tế, điều kiện sử dụng (dataset, cấu hình máy), và độ phù hợp với dataset Expedia.

### 3.1. LightGCL – mô hình nền tảng (baseline, ICLR 2023)

**Bước 1 – Xây dựng graph**

- Đầu vào: ma trận (user, hotel) mà user đó đã đặt phòng.
- Xử lý: biến ma trận này thành graph: mỗi user và mỗi hotel là một "điểm" (node), mỗi lượt đặt phòng là một "đường nối" (edge) giữa user đó và hotel đó. Sau đó chuẩn hoá lại mức độ ảnh hưởng của từng đường nối, để những user/hotel có quá nhiều kết nối (ví dụ một hotel siêu nổi tiếng được hàng nghìn người đặt) không "lấn át" những user/hotel ít kết nối hơn.
- Đầu ra: một graph đã chuẩn hoá, sẵn sàng để lan truyền thông tin qua.

**Bước 2 – Lan truyền (Propagation)**

- Đầu vào: graph từ Bước 1, cộng với một bảng embedding khởi tạo ngẫu nhiên (mỗi user/hotel được gán một vector số ngẫu nhiên ban đầu, coi như "chưa biết gì" về sở thích).
- Xử lý: với mỗi user, lấy trung bình có trọng số embedding của tất cả hotel mà user đó từng đặt, để cập nhật lại embedding của user đó — làm tương tự theo chiều ngược lại cho hotel (lấy trung bình embedding của các user đã đặt hotel đó). Lặp lại thao tác này qua nhiều layer, thường 2–4 layer — mỗi layer giúp thông tin lan xa hơn một bước trong graph, giống như hỏi bạn bè, rồi hỏi bạn của bạn bè.
- Đầu ra: sau mỗi layer, có một phiên bản embedding mới cho mỗi user/hotel. Cộng gộp tất cả các phiên bản qua các layer lại để ra một bảng embedding "lan truyền", mang thông tin từ tương tác trực tiếp lẫn gián tiếp.

**Bước 3 – Thêm nhánh đối chiếu qua SVD (Contrastive branch)**

- Đầu vào: cùng ma trận tương tác ở Bước 1. Các phương pháp contrastive learning trên graph thường tạo "phiên bản thứ hai" của graph bằng cách xoá ngẫu nhiên một số node/edge (random augmentation) – cách này dễ làm mất cấu trúc quan trọng và đưa nhiễu vào.
- Xử lý: LightGCL thay việc xoá ngẫu nhiên bằng SVD (Singular Value Decomposition) – một kỹ thuật phân rã ma trận tương tự PCA, giữ lại "q thành phần lớn nhất" (singular values lớn nhất) của ma trận tương tác, coi như giữ lại xu hướng chính và bỏ bớt chi tiết nhiễu. Không cần hiểu công thức đầy đủ, chỉ cần hiểu: đây là bước "nén rồi tái tạo lại" ma trận theo cách giữ thông tin quan trọng nhất. Graph "toàn cục, đã lọc sạch" này được lan truyền qua các bước như Bước 2 để ra một bảng embedding thứ hai.
- Đầu ra: hai bảng embedding song song cho cùng user/hotel – một từ graph gốc (Bước 2), một từ graph SVD (Bước 3).

**Bước 4 – Dự đoán & Huấn luyện**

- Đầu vào: hai bảng embedding từ Bước 2–3, cộng các cặp (user, hotel đã đặt) làm ví dụ đúng (positive) và một hotel chưa từng đặt làm ví dụ sai (negative).
- Xử lý: huấn luyện đồng thời hai mục tiêu – (1) loss xếp hạng: so khớp similarity giữa embedding user/hotel sao cho điểm của cặp đúng luôn cao hơn cặp sai; (2) loss đối chiếu (contrastive): kéo embedding của cùng một user/hotel ở 2 view (gốc và SVD) lại gần nhau, đồng thời đẩy embedding của các user/hotel khác nhau ra xa nhau.
- Đầu ra: bảng embedding đã học được (kết hợp cả hai view), dùng lại mỗi khi cần suy ra gợi ý mới cho một user.

**Điểm mạnh & điểm yếu**: mạnh ở chỗ không cần dò hyperparameter cho việc random augmentation (SVD tính một lần, xác định), và paper gốc cho thấy cải thiện Recall@20 khoảng 8–16% so với các phương pháp contrastive learning khác trên Yelp/Gowalla; đặc biệt bền vững trước dữ liệu thưa và thiên lệch phổ biến (popularity bias) – đúng hai vấn đề đã nêu ở mục 1. Điểm yếu: phải tính lại SVD khi ma trận tương tác thay đổi đáng kể (không cập nhật tăng dần được), và số chiều giữ lại q là một hyperparameter cần tinh chỉnh theo từng bộ dữ liệu.

**Giới hạn & khả năng ứng dụng thực tế**: SVD là một phép biến đổi mang tính toàn cục, nên với những user gần như cô lập (rất ít kết nối, đặc trưng của cold-start), "phiên bản graph toàn cục" này không mang lại nhiều thông tin bổ sung cho riêng họ.

**Điều kiện sử dụng**: cần ma trận tương tác user–item (implicit), một GPU tiêu chuẩn để huấn luyện embedding + contrastive loss, và thư viện hỗ trợ SVD xấp xỉ trên ma trận thưa (ví dụ randomized SVD); paper gốc thử nghiệm ở quy mô tới ~7 triệu tương tác, tương đương độ lớn dữ liệu clickstream của Expedia; có mã nguồn chính thức công khai (HKUDS/LightGCL).

**Độ phù hợp với Expedia**: ma trận user × hotel_cluster của Expedia rất thưa và có thiên lệch phổ biến rõ rệt (một số hotel_cluster được đặt nhiều hơn hẳn) – đúng là bối cảnh LightGCL nhắm tới; tuy nhiên với 58,43% user cold-start (gần như chỉ có 1 tương tác), lợi ích của "view toàn cục" sẽ hạn chế cho phần lớn user này, và đây cũng chính là lý do mục 3.5 đề xuất tiếp tục bổ sung Negative Sampling/denoising cho nhóm cold-start ở các bước sau.

**Lý do chọn LightGCL làm baseline** (thay vì nhảy thẳng lên RaDAR/SCCF):

- Cơ chế lan truyền ở Bước 1–2 vốn đơn giản (không dùng ma trận trọng số hay hàm phi tuyến), rẻ về chi phí tính toán, và là kiểu công thức lan truyền được nhiều nghiên cứu Graph CF gần đây dùng làm nền tảng ổn định.
- Nhánh đối chiếu qua SVD ở Bước 3–4 giải quyết trực tiếp hai vấn đề mà chính EDA trên Expedia đã chỉ ra ở mục 1 (ma trận cực kỳ thưa, feedback tích cực hiếm) – riêng phần lan truyền ở Bước 1–2 không có cơ chế nào xử lý việc này.
- Đã được kiểm chứng rộng rãi (ICLR 2023, hàng trăm trích dẫn, mã nguồn chính thức ổn định) – rủi ro triển khai thấp hơn nhiều so với RaDAR (2026, rất mới) hay SCCF (2024, khung lý thuyết còn ít được triển khai quy mô lớn).
- Chi phí tính SVD một lần rẻ hơn đáng kể so với việc huấn luyện thêm mô hình diffusion như RaDAR.

### 3.2. Các thuật toán Graph-based khác nghiên cứu để so sánh

#### 3.2.1. SCCF – Simple Contrastive Collaborative Filtering (KDD 2024)

- **Ý tưởng**: paper này đặt câu hỏi "vì sao contrastive learning lại hoạt động tốt trên graph CF?", và chứng minh rằng việc cập nhật embedding theo contrastive loss thực chất tương đương với hai phép graph convolution diễn ra đồng thời: một phép "kéo lại gần" theo cặp positive (giống hệt việc lan truyền/làm mượt embedding nhiều lớp như ở Bước 1–2 của LightGCL, mục 3.1), và một phép "đẩy ra xa" theo cặp negative (chống việc mọi embedding co cụm về một điểm). Vì contrastive loss đã ngầm làm luôn việc "lan truyền", SCCF bỏ hẳn các lớp graph propagation nhiều tầng, chỉ dùng một bảng embedding đơn giản cộng với một hàm contrastive loss được thiết kế lại.
- **Nền tảng toán học cơ bản**: không cần công thức – điểm mấu chốt cần nhớ là "cặp positive kéo gần = làm mượt = giống lan truyền graph; cặp negative đẩy xa = chống co cụm", nên một bảng embedding phẳng cộng contrastive loss có thể thay thế nhiều lớp GCN.
- **Điểm mạnh & điểm yếu**: đơn giản và rẻ hơn hẳn các mô hình lan truyền nhiều lớp như LightGCL (không có bước lan truyền nào cả → huấn luyện nhanh hơn, ít hyperparameter về số layer hơn); thực nghiệm trên Amazon-Beauty, Gowalla, Yelp2018, Pinterest cho kết quả ngang bằng hoặc tốt hơn các baseline dạng graph-convolution; tránh được vấn đề over-smoothing khi xếp chồng quá nhiều layer. Điểm yếu: vì bỏ hẳn lan truyền nhiều bước, có thể mất một phần tín hiệu gián tiếp (kiểu "bạn của bạn") mà các phương pháp xếp nhiều layer như LightGCL/RaDAR nắm bắt trực tiếp được.
- **Giới hạn & khả năng ứng dụng thực tế**: là một khung lý thuyết khá mới (2024), số lượng triển khai thực tế quy mô lớn còn ít hơn so với LightGCL.
- **Điều kiện sử dụng**: cùng loại input (ma trận tương tác), nhưng nhẹ hơn LightGCL về GPU/bộ nhớ vì không có bước propagation; paper dùng embedding size 64 trên nền RecBole – một cấu hình huấn luyện tiêu chuẩn, không đòi hỏi phần cứng đặc biệt.
- **Độ phù hợp với Expedia**: vì không cần propagation, SCCF là phương án rẻ nhất trong các thuật toán Graph-based khảo sát để thử nghiệm nhanh trên ma trận user × hotel_cluster – dùng như một "phép thử nhanh" xem riêng contrastive learning đã nắm được bao nhiêu tín hiệu hữu ích, đối chiếu với kết quả của baseline LightGCL đã chọn ở mục 3.1.

#### 3.2.2. RaDAR – Relation-aware Diffusion-Asymmetric Graph Contrastive Learning (WWW 2026)

- **Ý tưởng**: kết hợp một mô hình sinh dạng diffusion (thêm nhiễu dần rồi học cách khử nhiễu – như diffusion model trong sinh ảnh, nhưng áp dụng cho cấu trúc graph) với một bước "làm sạch quan hệ" (relation-aware) để tính lại trọng số các đường nối dựa trên đặc trưng ngữ nghĩa của node, sau đó đối chiếu (contrastive) hai nhánh không đối xứng – một nhánh "online" đang học, một nhánh "target" ổn định hơn – bằng cách lấy mẫu âm toàn cục (so sánh với mọi node trong batch chứ không chỉ vài mẫu).
- **Nền tảng toán học cơ bản**: không cần công thức – "diffusion" ở đây hiểu đơn giản là quá trình thêm nhiễu từng bước nhỏ rồi học một mạng nơ-ron để đảo ngược quá trình đó, tái tạo lại một graph "sạch" hơn bản gốc.
- **Điểm mạnh & điểm yếu**: giải quyết trực diện 2 vấn đề mà paper nêu ra – (1) augmentation ngẫu nhiên làm hỏng cấu trúc quan trọng, (2) các phương pháp cũ chỉ nắm được liên kết trực tiếp (1-hop), bỏ sót các user tương tự nhau qua liên kết gián tiếp (2-hop). Trên 3 benchmark công bố trong paper (Last.FM, Yelp, BeerAdvocate), RaDAR vượt qua toàn bộ baseline được so sánh trong paper (các phương pháp Graph CF truyền thống và contrastive learning khác) khoảng 3–5% Recall/NDCG (lưu ý: paper không trực tiếp so sánh với LightGCL). Điểm yếu: phức tạp nhất trong các phương pháp Graph-based khảo sát (3 thành phần: diffusion, làm sạch quan hệ, contrastive bất đối xứng phối hợp với nhau), tốn thêm chi phí tính toán cho bước khử nhiễu và bước đối chiếu toàn cục.
- **Giới hạn & khả năng ứng dụng thực tế**: là paper rất mới (công bố 2026), nên dù có mã nguồn công khai, số lượng kiểm chứng độc lập từ cộng đồng còn hạn chế so với LightGCL (2023).
- **Điều kiện sử dụng**: cùng input dạng ma trận user–item; khuyến nghị dùng GPU vì có thêm bước diffusion và bước đối chiếu toàn cục theo batch (chi phí tăng theo bình phương kích thước batch); cần tinh chỉnh batch size để cân bằng giữa chất lượng và bộ nhớ GPU; có mã nguồn tham khảo công khai (DGL + PyTorch).
- **Độ phù hợp với Expedia**: mục tiêu chính của RaDAR – khử nhiễu graph thưa và nắm bắt tương đồng gián tiếp (2-hop) – khớp khá sát với đặc điểm dữ liệu Expedia (ma trận đặt phòng thưa, tín hiệu hotel_cluster chỉ mang tính suy diễn chứ không phải rating tường minh); tuy nhiên vì chi phí và độ phức tạp cao hơn, và chưa có bằng chứng so sánh trực tiếp với LightGCL, đây phù hợp làm bước nâng cấp sau trên nền LightGCL (đúng như Bước 3 trong lộ trình mở rộng ở mục 3.5) hơn là baseline triển khai đầu tiên.

### 3.3. Hai thuật toán Session-based dùng để so sánh

#### 3.3.1. SASRec / TiSASRec

- **Ý tưởng**: coi lịch sử tương tác của mỗi user là một chuỗi theo thời gian, rồi dùng kiến trúc self-attention kiểu Transformer (giống mô hình dự đoán "từ tiếp theo" trong NLP, nhưng áp dụng cho chuỗi sản phẩm) để dự đoán item tiếp theo. Mỗi vị trí trong chuỗi chỉ được "nhìn" các vị trí trước nó (causal masking), nên mô hình có thể tự học nên chú ý nhiều vào hành động gần đây hay hành động cũ. TiSASRec mở rộng thêm: đưa thẳng khoảng cách thời gian thực tế giữa các lượt tương tác vào cơ chế attention, thay vì chỉ dùng thứ tự – "đặt phòng 3 ngày trước" và "đặt phòng 3 tháng trước" sẽ được xử lý khác nhau dù cùng thứ tự.
- **Nền tảng toán học cơ bản**: self-attention chuẩn (query/key/value, softmax trên điểm tương đồng) xếp chồng nhiều lớp; vị trí được mã hoá bằng positional embedding (SASRec) hoặc time-interval embedding (TiSASRec). Không cần đi sâu công thức attention để hiểu ý tưởng.
- **Điểm mạnh & điểm yếu**: nắm tốt "ý định ngắn hạn" trong phiên, không cần xây graph nên pipeline dữ liệu nhẹ hơn hẳn Graph-based; là baseline sequential được dùng phổ biến nhất, có rất nhiều mã nguồn ổn định (RecBole,…) nên rủi ro triển khai thấp; TiSASRec nắm bắt được thời gian trôi qua thực tế – hữu ích cho domain du lịch. Điểm yếu: cần một chuỗi tương tác *thực sự có thể học được*, tức user phải có nhiều hơn một vài lượt tương tác theo đúng thứ tự thời gian; nếu chỉ có 1–2 tương tác, attention gần như không có gì để "nhìn".
- **Giới hạn & khả năng ứng dụng thực tế**: chi phí tính toán O(n²·d) theo độ dài chuỗi không phải vấn đề ở quy mô phiên e-commerce thông thường; giới hạn thực sự nằm ở phía dữ liệu chứ không phải phía tính toán.
- **Điều kiện sử dụng**: cần log tương tác theo đúng thứ tự thời gian cho từng user (TiSASRec cần thêm khoảng cách thời gian thực tế, không chỉ thứ tự); GPU ở mức vừa phải vì chuỗi tương tác thường ngắn; có thể dùng ngay stack Transformer/RecBole tiêu chuẩn.
- **Độ phù hợp với Expedia**: đây là chỗ các phát hiện từ EDA thể hiện rõ nhất – như đã nêu ở đầu mục 3, 58,43% user của Expedia là cold-start (gần như chỉ 1 tương tác) và tỷ lệ session dao động 7,97%–25,78% tùy cách định nghĩa, nghĩa là phần lớn user không có đủ chuỗi tương tác để SASRec/TiSASRec học được gì. Kết quả này tái khẳng định bằng một kiến trúc cụ thể, được kiểm chứng rộng rãi, đúng kết luận định tính đã nêu: Session-based chưa phải hướng chính phù hợp cho Expedia ở giai đoạn này, dù TiSASRec (nhờ mô hình hoá khoảng cách thời gian) đáng cân nhắc lại sau này cho nhóm nhỏ user có nhiều lượt đặt phòng lặp lại.

#### 3.3.2. TRON – Transformer Recommender using Optimized Negative-sampling (RecSys 2023)

- **Ý tưởng**: TRON là một mô hình session-based hoàn chỉnh, xây trực tiếp trên kiến trúc SASRec (mục 3.3.1), do đội ngũ kỹ thuật của sàn thương mại điện tử OTTO phát triển. Thay vì chỉ lấy 1 mẫu âm ngẫu nhiên mỗi bước như SASRec gốc, TRON lấy nhiều mẫu âm cùng lúc theo 2 nguồn trộn lẫn – một phần lấy đều trên toàn bộ danh mục (uniform), một phần lấy theo tần suất xuất hiện thực tế của item (để mẫu âm "khó" hơn, giống các item hay được tương tác) – rồi dùng toàn bộ tập mẫu âm đó để tính loss dạng listwise (so sánh 1 lần với nhiều lựa chọn) thay vì chỉ so sánh từng cặp một.
- **Nền tảng toán học cơ bản**: phần self-attention giữ nguyên như SASRec (mục 3.3.1); phần mới là hàm loss – TRON dùng chính kỹ thuật **Sampled Softmax** đã giới thiệu ở mục 2.3 làm loss dạng listwise, thay cho binary cross-entropy hay BPR mà SASRec gốc hay dùng.
- **Điểm mạnh & điểm yếu**: đây là bằng chứng thực tế, đã triển khai production, cho thấy đúng ý tưởng "Negative Sampling nâng cao + Sampled Softmax" ở mục 2 mang lại cải thiện thật: trên Diginetica, TRON đạt Recall@20 = 0,537 so với SASRec gốc chỉ 0,454 (và vẫn vượt cả SASRec khi tự thêm BPR-Max hoặc Sampled Softmax riêng lẻ, ở mức 0,526/0,516); tốc độ huấn luyện gần tương đương SASRec (81 so với 94 epoch/giờ trên Diginetica) dù xử lý nhiều mẫu âm hơn; đã qua A/B test thật trên hệ thống OTTO, CTR tăng 18,14% so với SASRec. Điểm yếu: kế thừa nguyên vẹn giới hạn của SASRec – vẫn cần một chuỗi tương tác thực sự học được; 3 bộ dữ liệu benchmark của paper (Diginetica, Yoochoose, OTTO) đều có mật độ session dày hơn Expedia rất nhiều (riêng OTTO: 12,9 triệu session huấn luyện).
- **Giới hạn & khả năng ứng dụng thực tế**: giá trị chính của TRON nằm ở việc tối ưu tốc độ/độ chính xác khi danh mục lớn và có nhiều session – không giải quyết vấn đề gốc là *thiếu* session per user; cách lấy mẫu âm (theo từng bước, theo session, hay theo cả batch) là một lựa chọn thiết kế cần cân nhắc đánh đổi giữa chất lượng và bộ nhớ.
- **Điều kiện sử dụng**: cùng input như SASRec (log tương tác có thứ tự thời gian theo user/session); GPU ở mức tương đương SASRec vì tác giả tối ưu để giữ tốc độ huấn luyện gần như không đổi; có mã nguồn chính thức công khai (github.com/otto-de/TRON) và cả bộ dữ liệu ẩn danh đi kèm.
- **Độ phù hợp với Expedia**: vì TRON dùng chung backbone SASRec, nó thừa hưởng đúng giới hạn đã nêu ở mục 3.3.1 – 58,43% user cold-start và ranh giới session chưa rõ ràng (7,97%–25,78%) khiến một backbone sequential khó là lựa chọn chính hiện tại. Giá trị lớn nhất của TRON với dự án là minh chứng thực tế, đã qua A/B test, cho việc kết hợp Negative Sampling nâng cao + Sampled Softmax (đúng 2 kỹ thuật ở mục 2.1 và 2.3) – nên được dùng làm tham khảo trực tiếp khi triển khai Bước 2 và Bước 4 trong lộ trình mở rộng trên nền LightGCL ở mục 3.5, thay vì áp dụng nguyên bản kiến trúc sequential của nó.

### 3.4. Kết luận lựa chọn

Dựa trên kết quả nghiên cứu sâu 5 thuật toán ở mục 3, nhóm quyết định chọn **LightGCL** làm mô hình nền tảng (baseline), vì các lý do đã trình bày ở mục 3.1:

- Cơ chế lan truyền đơn giản (Bước 1–2, mục 3.1) – không dùng ma trận trọng số hay hàm phi tuyến – rẻ về chi phí tính toán và đã được nhiều nghiên cứu Graph CF gần đây dùng làm nền tảng ổn định.
- Nhánh contrastive learning qua SVD giải quyết trực tiếp đặc điểm ma trận cực kỳ thưa và feedback tích cực hiếm mà EDA đã chỉ ra ở mục 1 – điều phần lan truyền thuần không xử lý được.
- Đã được kiểm chứng rộng rãi (ICLR 2023, mã nguồn chính thức ổn định) – rủi ro triển khai thấp hơn RaDAR (2026, rất mới) hay SCCF (2024, còn ít triển khai quy mô lớn).

Hai thuật toán Graph-based khác khảo sát ở mục 3.2 (SCCF, RaDAR) không thay đổi lựa chọn trên, nhưng xác nhận lại nhận định ở mục 3 rằng phần kiến trúc lan truyền ít thay đổi qua các thế hệ – cả hai đều dùng cùng kiểu công thức lan truyền đơn giản mà LightGCL cũng dùng làm nền; cải tiến của chúng nằm ở cách tạo/làm sạch graph (RaDAR) và cách thiết kế loss (SCCF, RaDAR) – hai hướng được đưa vào lộ trình mở rộng ở mục 3.5 thay vì thay thế LightGCL ngay từ đầu.

Hai thuật toán Session-based khảo sát ở mục 3.3 (SASRec/TiSASRec, TRON) cũng không thay đổi kết luận trên – ngược lại, chúng xác nhận cụ thể hơn lý do Session-based chỉ nên dùng để so sánh ở giai đoạn này: tỷ lệ cold-start 58,43% và ranh giới session chưa rõ ràng (7,97%–25,78% tuỳ định nghĩa) khiến một kiến trúc sequential kinh điển như SASRec (kể cả bản TRON đã tối ưu) khó có đủ tín hiệu để học, dù về lâu dài các ý tưởng cụ thể của chúng (mô hình hoá khoảng thời gian của TiSASRec, tổ hợp Negative Sampling + Sampled Softmax đã được TRON kiểm chứng bằng A/B test thật) vẫn có thể mượn lại cho pipeline Graph-based.

### 3.5. Lộ trình mở rộng từ baseline

Từ những phân tích trên, ta sẽ chọn LightGCL là điểm khởi đầu để từng bước bổ sung các cải tiến mới nhất. Lộ trình thử nghiệm được đề xuất đi từ đơn giản đến phức tạp, mỗi bước chỉ bổ sung đúng một kỹ thuật mới để có thể so sánh công bằng với bước trước đó:

- **Bước 1**: LightGCL (đã bao gồm lan truyền graph + nhánh contrastive learning qua SVD, mục 3.1) không áp dụng cải tiến gì thêm, dùng làm baseline.
- **Bước 2**: Kết hợp Negative Sampling nâng cao – thay vì chọn negative sample ngẫu nhiên, áp dụng chiến lược lấy mẫu đa nguồn (trộn uniform và theo tần suất phổ biến) như TRON ở mục 3.3.2 để mô hình học phân biệt tốt hơn.
- **Bước 3**: Bổ sung thêm kỹ thuật denoising/làm sạch graph nằm ngoài phần SVD sẵn có của LightGCL – làm sạch và lọc bớt các liên kết không đáng tin cậy trong dữ liệu trước khi đưa vào propagation (tham khảo hướng khử nhiễu bằng diffusion của RaDAR, hoặc đối chiếu với góc nhìn contrastive-only của SCCF, ở mục 3.2).
- **Bước 4**: Bổ sung thêm Sampled Softmax cải tiến – thay hàm mất mát cuối cùng bằng phiên bản tối ưu hơn (như loss listwise mà TRON dùng ở mục 3.3.2, đã được kiểm chứng bằng A/B test thật), kết hợp toàn bộ các cải tiến ở các bước trước.

Mỗi bước sẽ được huấn luyện và đánh giá độc lập trên cùng một bộ dữ liệu, với cùng một cách chia tập huấn luyện và tập kiểm tra, nhằm đảm bảo việc so sánh là công bằng và phản ánh đúng đóng góp thực sự của từng kỹ thuật.

### 3.6. Đánh giá và so sánh kết quả

Ở mỗi bước trong lộ trình, cần đo các chỉ số đánh giá tiêu chuẩn của hệ gợi ý, ví dụ Recall@K và NDCG@K, dùng để đo mức độ chính xác của danh sách top-K gợi ý được đưa ra, trên cùng một tập dữ liệu kiểm tra. Việc đo lường nhất quán này cho phép so sánh công bằng hiệu quả của từng kỹ thuật được bổ sung so với mô hình nền tảng LightGCL ban đầu, từ đó xác định rõ đóng góp thực sự của mỗi cải tiến.

## 4. Kết luận

Phương pháp luận trình bày trong tài liệu này xuất phát từ việc bắt đầu bằng LightGCL – một mô hình kết hợp cơ chế lan truyền graph đơn giản, rẻ về chi phí tính toán, với một nhánh contrastive learning dựa trên SVD giải quyết trực tiếp đặc điểm thưa và ít feedback tích cực của dữ liệu Expedia – sau đó từng bước bổ sung các kỹ thuật đã được chứng minh hiệu quả trong các nghiên cứu tiên tiến gần đây, bao gồm negative sampling nâng cao, denoising graph, và sampled softmax. Cách tiếp cận theo từng bước này vừa đảm bảo tính khả thi khi triển khai thực tế, vừa cho phép đánh giá rõ ràng đóng góp của từng kỹ thuật khi áp dụng lên dữ liệu đặt phòng khách sạn.
