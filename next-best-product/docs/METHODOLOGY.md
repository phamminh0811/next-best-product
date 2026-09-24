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

## 3. Họ phương pháp Graph-based Collaborative Filtering

Dù cả hai hướng đều khả thi với bộ dữ liệu Expedia, dự án quyết định phát triển sâu theo hướng Graph-based trước vì:

- Định nghĩa session trên dữ liệu Expedia chưa thực sự rõ ràng: kết quả EDA cho thấy các cách định nghĩa session khác nhau (theo ngày, theo khoảng nghỉ 30 phút, theo destination) cho ra tỷ lệ chốt rất khác nhau, từ 7,97% đến 25,78%, nghĩa là cần thêm công sức tiền xử lý và thử nghiệm để chọn đúng định nghĩa trước khi có thể huấn luyện một mô hình Session-based đáng tin cậy.
- Ma trận user × hotel_cluster để làm đã có sẵn và rõ ràng ngay từ dữ liệu gốc, không cần thêm bước xử lý trung gian, giúp rút ngắn thời gian từ nghiên cứu đến có baseline chạy được.

Trong hướng Graph-based Collaborative Filtering, các nghiên cứu qua từng giai đoạn đã đi theo một quá trình phát triển khá rõ rệt. Giai đoạn đầu tập trung đơn giản hoá kiến trúc mạng nơ-ron để việc huấn luyện trở nên dễ dàng, nhanh và ổn định hơn. Giai đoạn sau, tập trung làm giàu chất lượng huấn luyện bằng các kỹ thuật self-supervised learning và contrastive learning, đặc biệt là các phương pháp denoising graph – tức lọc bớt những mối liên kết không đáng tin cậy hoặc bù đắp những mối liên kết còn thiếu trong dữ liệu quan sát được, trước khi đưa vào mô hình.

Một điểm đáng chú ý là các nghiên cứu càng về sau càng cho thấy phần "kiến trúc lan truyền", tức cách mạng nơ-ron xử lý đồ thị, hầu như không còn thay đổi nhiều so với giai đoạn đầu – phần lớn vẫn dựa trên công thức lan truyền tuyến tính đơn giản đã được chứng minh hiệu quả từ sớm. Sự cải tiến trong các nghiên cứu mới chủ yếu đến từ hai chỗ khác: cách xây dựng và làm sạch đồ thị đầu vào, và cách thiết kế hàm mất mát, đặc biệt là cách lấy mẫu âm và áp dụng contrastive learning.

### 3.2. Cách Graph-based Collaborative Filtering hoạt động - LightGCN

**Bước 1 – Xây dựng graph**

- Đầu vào: ma trận (user, hotel) mà user đó đã đặt phòng.
- Xử lý: biến ma trận này thành graph: mỗi user và mỗi hotel là một "điểm" (node), mỗi lượt đặt phòng là một "đường nối" (edge) giữa user đó và hotel đó. Sau đó chuẩn hoá lại mức độ ảnh hưởng của từng đường nối, để những user/hotel có quá nhiều kết nối (ví dụ một hotel siêu nổi tiếng được hàng nghìn người đặt) không "lấn át" những user/hotel ít kết nối hơn.
- Đầu ra: một graph đã chuẩn hoá, sẵn sàng để lan truyền thông tin qua.

**Bước 2 – Lan truyền (Propagation)**

- Đầu vào: graph từ Bước 1, cộng với một bảng embedding khởi tạo ngẫu nhiên (mỗi user/hotel được gán một vector số ngẫu nhiên ban đầu, coi như "chưa biết gì" về sở thích).
- Xử lý: với mỗi user, lấy trung bình có trọng số embedding của tất cả hotel mà user đó từng đặt, để cập nhật lại embedding của user đó — làm tương tự theo chiều ngược lại cho hotel (lấy trung bình embedding của các user đã đặt hotel đó). Lặp lại thao tác này qua nhiều layer, thường 2–4 layer — mỗi layer giúp thông tin lan xa hơn một bước trong graph, giống như hỏi bạn bè, rồi hỏi bạn của bạn bè.
- Đầu ra: sau mỗi layer, có một phiên bản embedding mới cho mỗi user/hotel. Cộng gộp tất cả các phiên bản qua các layer lại để ra một bảng embedding cuối cùng, mang thông tin từ tương tác trực tiếp lẫn gián tiếp.

**Bước 3 – Dự đoán (Prediction)**

- Đầu vào: embedding cuối cùng của một user và một hotel, lấy từ Bước 2.
- Xử lý: so khớp similarity giữa hai vector này.
- Đầu ra: điểm dự đoán mức độ user đó sẽ thích hotel đó.

**Bước 4 – Huấn luyện (Training)**

- Đầu vào: các cặp (user, hotel đã đặt) làm ví dụ đúng (positive), và một hotel ngẫu nhiên mà user đó chưa từng đặt làm ví dụ sai (negative).
- Xử lý: chỉnh dần bảng embedding ban đầu (dùng ở Bước 2) sao cho điểm số (Bước 3) của cặp đúng luôn cao hơn điểm số của cặp sai.
- Đầu ra: bảng embedding đã học được, dùng lại ở Bước 2–3 mỗi khi cần suy ra gợi ý mới cho một user.

### 3.3. Lý do chọn LightGCN làm mô hình nền tảng (baseline)

Dựa trên quá trình phát triển nêu trên, nhóm quyết định chọn LightGCN làm mô hình nền tảng, vì các lý do sau đây:

- Đơn giản nhất trong họ phương pháp: LightGCN chỉ giữ lại đúng một thao tác cốt lõi – lan truyền và cộng gộp thông tin qua các lớp của graph – và loại bỏ mọi thành phần không thật sự cần thiết.
- Đã được kiểm chứng rộng rãi: đây là một trong những mô hình được trích dẫn và tái sử dụng nhiều nhất trong lĩnh vực hệ gợi ý, có nhiều bộ mã nguồn tham khảo chất lượng cao, giúp giảm rủi ro khi triển khai lần đầu.
- Là nền tảng cho hầu hết các phương pháp mới hơn: nhiều nghiên cứu tiên tiến thực chất vẫn sử dụng lại đúng công thức lan truyền của LightGCN làm phần lõi, rồi thêm các lớp cải tiến bên ngoài.

### 3.4. Lộ trình mở rộng từ baseline

Từ những phân tích trên, ta sẽ chọn LightGCN là điểm khởi đầu để từng bước bổ sung các cải tiến mới nhất. Lộ trình thử nghiệm được đề xuất đi từ đơn giản đến phức tạp, mỗi bước chỉ bổ sung đúng một kỹ thuật mới để có thể so sánh công bằng với bước trước đó:

- **Bước 1**: LightGCN không áp dụng cải tiến nào, dùng làm baseline.
- **Bước 2**: LightGCN kết hợp Negative Sampling nâng cao – thay vì chọn negative sample ngẫu nhiên, áp dụng chiến lược lấy mẫu có chọn lọc để mô hình học phân biệt tốt hơn.
- **Bước 3**: Bổ sung thêm kỹ thuật denoising graph – làm sạch và lọc bớt các liên kết không đáng tin cậy trong dữ liệu trước khi đưa vào propagation.
- **Bước 4**: Bổ sung thêm Sampled Softmax cải tiến – thay hàm mất mát cuối cùng bằng phiên bản tối ưu hơn, kết hợp toàn bộ các cải tiến ở các bước trước.

Mỗi bước sẽ được huấn luyện và đánh giá độc lập trên cùng một bộ dữ liệu, với cùng một cách chia tập huấn luyện và tập kiểm tra, nhằm đảm bảo việc so sánh là công bằng và phản ánh đúng đóng góp thực sự của từng kỹ thuật.

### 3.5. Đánh giá và so sánh kết quả

Ở mỗi bước trong lộ trình, cần đo các chỉ số đánh giá tiêu chuẩn của hệ gợi ý, ví dụ Recall@K và NDCG@K, dùng để đo mức độ chính xác của danh sách top-K gợi ý được đưa ra, trên cùng một tập dữ liệu kiểm tra. Việc đo lường nhất quán này cho phép so sánh công bằng hiệu quả của từng kỹ thuật được bổ sung so với mô hình nền tảng LightGCN ban đầu, từ đó xác định rõ đóng góp thực sự của mỗi cải tiến.

## 4. Kết luận

Phương pháp luận trình bày trong tài liệu này xuất phát từ việc bắt đầu bằng một mô hình đơn giản và đã được kiểm chứng rộng rãi, là LightGCN, sau đó từng bước bổ sung các kỹ thuật đã được chứng minh hiệu quả trong các nghiên cứu tiên tiến gần đây, bao gồm negative sampling nâng cao, denoising graph, và sampled softmax. Cách tiếp cận theo từng bước này vừa đảm bảo tính khả thi khi triển khai thực tế, vừa cho phép đánh giá rõ ràng đóng góp của từng kỹ thuật khi áp dụng lên dữ liệu đặt phòng khách sạn.
