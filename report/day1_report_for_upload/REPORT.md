# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (T4)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (giữ nguyên, không đổi)

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Trong JSON chỉ có một lớp, một thứ hạng và một score cho ảnh `traffic`; không có `bbox_xyxy` hay `polygon_xy` nên không chỉ vào vật thể cụ thể nào, cũng không cho biết vật nằm ở đâu.
3. Danh sách lớp **đi kèm taxonomy/checkpoint** (ImageNet-1K, 1000 lớp) mà `yolo11n-cls.pt` được huấn luyện. Model chỉ chọn và xếp hạng các lớp trong danh sách có sẵn, không tự nghĩ ra lớp mới. Vì ảnh mẫu là cảnh COCO nhưng taxonomy là ImageNet nên top-1 có thể trông “lạ”; đó là hệ quả của việc taxonomy không khớp nội dung ảnh, không phải lỗi máy.
4. `class_id` là mã số, `class_name` là tên đọc được, `taxonomy_name` cho biết bảng tra nào sinh ra cả hai. Cùng một số có thể ứng với lớp khác nhau ở taxonomy khác (ImageNet-1K và COCO-80 đánh số khác nhau), và cùng một tên có thể mang số khác. Nếu chỉ ghi tên lớp thì không đối chiếu, không tái lập và không ánh xạ sang taxonomy khác được.
5. Guideline cần nói rõ: (a) tiêu chí chọn **một** lớp chính khi ảnh có nhiều vật/chủ đề (ví dụ vật hoặc chủ đề chiếm nhiều diện tích nhất / nổi bật nhất ở trung tâm); (b) thứ tự ưu tiên khi hai lớp ngang nhau; (c) khi nào dùng lớp “khác/không xác định” hoặc loại ảnh; (d) cấm gán nhiều lớp nếu tác vụ là single-label.
6. `score` là mức tin của **model**, không có ai xác minh. Model có thể rất tự tin mà vẫn sai (ví dụ do taxonomy không khớp cảnh ảnh). Ground truth phải do người gán theo guideline và qua QC; nếu lấy score làm ground truth thì lỗi của model sẽ trở thành “nhãn đúng” và bị đưa ngược vào huấn luyện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Record chọn: `class_name`=bowl ; `score`=0.701282 ; `bbox_xyxy`=[
   155.02,
   168.67,
   182.38,
   184.18
   ]; `bbox_width`=27.36 ; `bbox_height`=15.51
2. `bbox_xyxy` = `[x_min, y_min, x_max, y_max]` theo pixel, gốc `(0, 0)` ở góc trên-trái ảnh, x tăng sang phải, y tăng xuống dưới. Với record trên: góc trên-trái ở (`x_min`, `y_min`) = (155.02, 168.67), góc dưới-phải ở (`x_max`, `y_max`) = (182.38, 184.18); hộp rộng `x_max − x_min` = 27.36 px và cao `y_max − y_min` = 15.51 px.
3. Số prediction theo ngưỡng (lấy từ output ô so sánh trong notebook, ảnh `kitchen`): ngưỡng 0.20 → 17 prediction; ngưỡng 0.60 → 6 prediction. (Ngưỡng 0.35, dùng cho JSON: 11 prediction.)
4. Hạ ngưỡng thì giữ thêm nhiều hộp có score thấp, trong đó có nhiều hộp thừa, sai lớp hoặc lệch vị trí. Reviewer phải kiểm tra thêm từng hộp mới: đúng lớp không, có phải vật thật không, hộp ôm sát chưa, có trùng lặp với hộp khác không; rồi xóa hoặc sửa. Khối lượng review tăng, còn hạ ngưỡng quá thấp thì tỉ lệ hộp sai tăng.
5. **Quy tắc box:** vẽ một hộp chữ nhật sát khít quanh phần **nhìn thấy** của từng vật thể (không chừa viền thừa, không cắt mất phần vật), mỗi vật thể một hộp riêng, không gộp nhiều vật cùng lớp vào một hộp.
6. Vật bị che khuất/cắt mép trên `kitchen`: Ở mép trái ảnh có một person (score 0.61) bị cắt ngang mép ảnh, chỉ lộ ra phần tay/vai; ngay cạnh còn có oven (0.69) và bowl (0.58) cũng bị cắt ở mép trái. Ngoài ra bowl (0.50) và cup (0.45) nằm chồng lên nhau trên bàn nên bị che một phần. **Cách xử lý:** viết thành quy tắc trong guideline, vì tình huống che khuất/cắt mép lặp lại ở rất nhiều ảnh nên cần một cách làm thống nhất (ví dụ: vật bị che thì chỉ vẽ phần nhìn thấy; vật bị cắt mép thì vẽ đến mép ảnh; vật nhìn thấy dưới một tỉ lệ nhất định thì bỏ qua). Chỉ escalation lên Lab Coach khi ca đó nằm ngoài các quy tắc đã có hoặc không thể quyết định chắc chắn, thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Record chọn: `instance_id`=kitchen-006 ; `class_name`=bowl ; `score`=0.528201 ; `polygon_point_count`=84 ; 3–5 điểm đầu của `polygon_xy`=[[29.0, 303.0], [28.0, 304.0], [26.0, 304.0]]
2. Box chỉ cho biết vùng chữ nhật chứa vật. Polygon bám theo **đường biên thật** của vật, nên cho biết hình dạng, phần diện tích vật thực sự chiếm và phần nền nằm trong hộp nhưng không thuộc vật. Từ polygon có thể tính diện tích, tách vật khỏi nền và tách hai vật nằm gần nhau.
3. `instance_id` (dạng `kitchen-001`) chỉ là số thứ tự để phân biệt từng instance **trong output của bài lab**: hai vật cùng lớp vẫn có hai `instance_id` và hai polygon riêng. Nó **không phải class ID** vì không cho biết vật thuộc lớp nào (hai instance cùng lớp có ID khác nhau, còn `class_id` thì giống nhau). Nó **không phải tracking ID** vì chỉ áp dụng trong một ảnh tĩnh, không theo dõi cùng một vật qua nhiều khung hình video.
4. **Quy tắc vẽ biên mask:** đặt điểm polygon bám sát đường biên ngoài của phần nhìn thấy của vật, đủ dày ở chỗ cong để không cắt góc, không lấn sang nền hoặc vật bên cạnh; mỗi vật một mask riêng và các mask không chồng lên nhau.
5. Chỗ khó trên `kitchen`: Ở góc trái phía dưới ảnh kitchen, nhiều instance chạm và chồng lên nhau: mask oven (0.49, màu tím) kéo xuống đè lên vùng dining table (0.58); mask dining table bao cả các vật đặt trên bàn (lọ, cây cán bột) nên biên của nó không rõ đâu là mặt bàn, đâu là vật; person (0.43) chỉ là một mảng nhỏ sát mép trái bị cắt, lẫn với các bowl (0.70 và 0.53) chồng lên nhau. Ở những chỗ này đường biên giữa các vật không rõ và nhiều annotator có thể vẽ khác nhau. **Cách xử lý:** dùng **guideline** nếu tình huống phổ biến, ví dụ quy định rõ đường chia giữa hai vật chạm nhau và cách xử lý phần bị che. Nếu chỗ đó mơ hồ đến mức nhiều annotator sẽ vẽ khác nhau và guideline chưa nói tới, thì **escalation** lên Lab Coach, rồi bổ sung kết quả vào guideline.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                                                      | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                                                                                                                                         | Annotator làm gì?                                                                                                                       | Reviewer xem gì?                                                                                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| Phân loại ảnh         | Một lớp (`class_id` + `class_name` + taxonomy) cho **cả ảnh**; đơn vị nhãn là ảnh                  | Ảnh `traffic` là cảnh đường phố có nhiều vật nhưng chỉ được một lớp; top-1 là `cab` (`class_id`=468, score 0.510915), chỉ phản ánh một phần cảnh và score chưa tới 0.6 nên không chắc chắn; taxonomy ImageNet không khớp cảnh COCO                                         | Chọn một lớp chính theo tiêu chí trong guideline, chọn “khác/không rõ” khi không khớp taxonomy, không dựa vào score của model           | Lớp có đúng theo guideline không, có chọn đúng vật/chủ đề chính không, có dùng đúng taxonomy và `class_id` không                                                  |
| Phát hiện vật thể     | Một **hộp** `[x_min, y_min, x_max, y_max]` (pixel) + lớp cho **mỗi vật thể**; đơn vị nhãn là hộp   | Ảnh `kitchen`: số hộp đổi mạnh theo ngưỡng (0.20 → 17, 0.35 → 11, 0.60 → 6 prediction); `person` (0.61), `oven` (0.69) và `bowl` (0.58) bị cắt ở mép trái ảnh; `bowl` (0.50) và `cup` (0.45) chồng lên nhau trên bàn                                                       | Vẽ hộp sát khít quanh phần nhìn thấy của từng vật, chọn đúng lớp COCO, vật bị cắt mép thì vẽ đến mép ảnh, đánh dấu ca khó để hỏi        | Hộp có sát không, có sót vật hoặc hộp thừa/trùng không (đặc biệt các hộp score thấp khi hạ ngưỡng), có đúng lớp không, ca che khuất/cắt mép có theo quy tắc không |
| Instance segmentation | Một **polygon** (các điểm pixel bám biên) + lớp cho **mỗi instance**; đơn vị nhãn là mask từng vật | Ảnh `kitchen`: mask `oven` (0.49) chồng lên `dining table` (0.58) ở góc trái, mask bàn bao cả vật đặt trên bàn, nhiều `bowl` chồng nhau; polygon `kitchen-006` (`bowl`, 0.528201) có 84 điểm và nằm trong vùng bowl chồng nhau nên biên dễ lấn sang bowl hoặc bàn bên cạnh | Vẽ biên bám sát phần nhìn thấy của từng instance, tách riêng các vật chạm nhau, mỗi vật một mask không chồng; escalation nếu biên mơ hồ | Biên có bám sát không, instance có bị gộp/tách sai không, mask có chồng lấn không, lớp đúng không                                                                 |     |
