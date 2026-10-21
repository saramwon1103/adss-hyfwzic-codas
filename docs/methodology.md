# Mode 2 thực: quyết định triển khai, dữ liệu và cách đọc kết quả

## Mục tiêu của thay đổi

Thay nguồn dữ liệu tổng hợp mặc định bằng file ZIP người dùng cung cấp. Không chỉ đổi tên hoặc gắn nhãn synthetic=False cho dữ liệu sinh ngẫu nhiên. Bộ dữ liệu thực được nạp từ CSV đã đóng kèm; thiếu source thì báo lỗi, không quay sang mock.

## 1. Chọn sector và khoảng thời gian

README nguồn xác định sector 1 và 2 là cà chua ngoài đồng; sector 3 là cà chua chậu, sector 4 bí ngòi, sector 5 việt quất. Bản 1.1 chỉ cho chọn 1/2 để giữ cùng bài toán tomato.

Mỗi sector có 33.259 hàng trên lưới 10 phút, từ 09/02/2025 đến 27/09/2025 23:00. Chỉ tổng hợp các ngày đủ 144 ô thời tiết; còn 230 ngày đầy đủ. Không xóa ngày chỉ vì sensor thiếu, không nối ngày xa nhau thành một transition 24 giờ.

Mặc định đánh giá 01/06–13/07/2025, 43 ngày, giai đoạn còn có hoạt động tưới và quan trắc ở hai sector. Dữ liệu sau đó có chuỗi độ ẩm gần hằng/giảm chậm và van ít hoặc không tưới; không lấy độ phủ sensor cao nhất để suy ra một thí nghiệm điều khiển tốt nhất.

Training lấy các cặp có cả sensor đầu và đầu ngày kế tiếp; **target cuối của training phải trước ngày bắt đầu đánh giá**. Không xáo trộn random train/test. Không chọn hệ số theo kết quả holdout.

## 2. Chọn lớp merged thay vì preprocessed

Dùng `02_processed_data/merged/dataset_zone_1.csv` và zone 2 trong ZIP. Lớp này giữ thời gian đều, weather, thời lượng van và liters_total. Tránh lớp preprocessed vì có các cột target tương lai, các dòng bị loại khiến một số cửa sổ đếm theo hàng không còn đúng thời gian thực và không còn liters_total.

Trong hai file merged, xác minh `liters_total = irrigation_duration_minutes × 1.1` trong sai số 1e-6. Đây là lượng nước **tái dựng bằng lưu lượng danh định**, không phải số đo của lưu lượng kế.

## 3. Đơn vị

- Không có diện tích tưới hữu hiệu: giữ L, không tự đổi sang mm bằng một diện tích tự đặt.
- Moisture là giá trị probe tương đối 0–100. Không chia 100 rồi tuyên bố là VWC m³/m³ đã hiệu chuẩn.
- MADP=40, FC=60 lấy từ TOMATO_SOIL trong config/README nguồn. Target=50 là giả định midpoint của bản mô phỏng.
- Không có ET0 trong CSV: không tạo ngẫu nhiên ET0 hoặc gọi direct radiation là tổng radiation để tính FAO-56 khi thiếu thông tin.

## 4. Tổng hợp thời tiết và sensor

- Temperature/RH: trung bình theo ngày local wall-clock.
- Radiation: tổng direct-radiation W/m² ×600 giây /10⁶ → MJ/m²; vẫn ghi rõ là direct-radiation proxy.
- Rain: source code lấy rain theo giờ từ API rồi nội suy và lấy trung bình lưới 10 phút. Tích phân xấp xỉ `sum(weather_rain × 10/60)`, tránh cộng trực tiếp thành lượng mưa cao gấp sáu lần. Đây không khôi phục chính xác hourly rain gốc và có thể sai lệch ở ranh giới giờ/ngày; report giữ cách tính.
- Soil: loại NaN và ngoài 0–100, không dùng centred rolling median. Chọn giá trị mới nhất tại/trước nửa đêm, tuổi không quá 12h. Giữ sensor_age_hours và sensor_observed_at để audit. Không dùng quan trắc sau thời điểm quyết định lấp về quá khứ.
- Không hứa đã loại hết spike nằm trong 0–100 hoặc đã phát hiện mọi flatline. Chất lượng sensor còn là giới hạn.
- Timestamp nguồn là Europe/Rome naive, không sửa DST trong file gốc. Daily key giữ wall-clock date của nguồn, không tạo độ chính xác UTC mà source không có.

## 5. Mô hình đáp ứng đất

Để thay lượng tưới mà không ép ba policy đi theo độ ẩm lịch sử của một lịch tưới khác, cần mô hình trạng thái. Bản 1.1 dùng constrained ridge daily:

`next_probe_index = intercept + b0*current_index + b1*irrigation_L + b2*temperature + b3*rain + b4*direct_radiation + b5*RH`.

Các feature được chuẩn hóa bằng mean/std **training**. Ridge alpha=10 cố định. Ràng buộc b0 trong [0,1], b1≥0; giải sáu tổ hợp active-set. Các hệ số weather không bị gán nghĩa nhân quả; sự liên hệ của chúng có thể chịu confounding. Output index giới hạn 0–100.

Mô hình này không phải ANFIS, không phải mô hình nước vật lý và không bảo đảm mass balance. Nếu b1 gần 0 thì không dựng hiệu ứng tưới giả; dose inversion trả 0 khi gain không xác định được. Khi gain yếu hoặc dự báo kém baseline, report/UI nêu rõ.

Trong evaluation, weather thực của ngày làm forcing của surrogate; strategy chỉ nhận dự báo persistence từ ngày trước. Sau thời điểm đầu, sensor mà policy nhìn thấy là trạng thái surrogate của chính policy đó theo lịch availability nguồn; không gắn quan trắc lịch sử vào kết quả của action mới. Không sinh nhiễu ngẫu nhiên để làm giả sensor thật.

Giới hạn khoa học: hồi quy quan sát không xác định được hiệu quả nhân quả của lượng nước vì người vận hành tưới theo trạng thái cây/đất; thời tiết, stage và sensor có thể đồng biến. Xếp hạng mô phỏng không thay cho thí nghiệm canh tác ngẫu nhiên hoặc mô hình đất đã hiệu chuẩn.

## 6. Đánh giá holdout

Một bước mỗi lần, dùng độ ẩm đầu ngày và lượng nước lịch sử của chính ngày để dự báo sensor đầu ngày sau, so với target giữ riêng. So với baseline `next=current`.

| Sector | Train pairs | Holdout pairs | MAE surrogate | MAE persistence |
|---|---:|---:|---:|---:|
| 1 | 87 | 39 | 7.4818 | 7.5897 |
| 2 | 85 | 27 | 6.8668 | 5.5556 |

Đơn vị là điểm probe. Đây là kiểm tra dự báo một bước khi biết water/weather lịch sử, không phải kiểm tra forecast thời tiết vận hành và không phải sai số của free-running counterfactual trajectories. Sector 1 chỉ hơn baseline rất ít; sector 2 kém baseline. Cả hai có feature khí tượng ngoài miền training. Mức thay đổi moisture từ một ngày nước ở daily cap còn thấp hơn sai số holdout. Report không che những kết quả này.

## 7. Kịch bản

- Normal: forcing từ source.
- Drought: từ step 10, temp +4°C, RH −15 điểm (chặn tại 0), direct radiation ×1.15, mưa ×0.1, daily water allocation ×0.6. Đây là what-if, không khẳng định một đợt hạn thật đã xảy ra.
- Missing Sensor: mất thêm sensor step 10..18, giữ missing tự nhiên ở mọi scenario.
- Forecast cho d sau biến đổi vẫn dùng forcing d−1; ngày đầu shock chưa được thấy trước qua forecast.

## 8. Năm KPI thực

| ID | Đơn vị | Công thức/ý nghĩa |
|---|---|---|
| water_use_liters | L | Tổng lượng tưới policy đề xuất |
| soil_moisture_mae_points | điểm index | mean(abs(modeled_index_end−50)), không phải MAE dự báo holdout |
| over_irrigation_liters | L proxy | Phần action vượt lượng nước ước tính cần để dự báo đạt upper=60; không phải drainage vật lý được đo |
| replanning_response_days | ngày | Sau demand_event, ngày đầu dose ≥ modeled feasible demand − max(0.1 L,5% cap), tối đa 7 ngày; không có event=0 |
| missing_data_robustness | 0..1 | clip(1−max(0,MAE_added_outage−MAE_natural_missing)/10,0,1) |

Robustness chạy cùng forcing/cap/policy và lịch thiếu tự nhiên; thêm mất sensor chỉ ở cặp stress. Điểm 1 có thể do policy bão hòa cap hoặc không nhạy thêm với missing, không chứng minh sensor không quan trọng. Normal/Missing Sensor mặc định sector 1 có thể cho matrix giống nhau; không thêm nhiễu hoặc đổi luật chỉ để tạo khác biệt.

Weights vẫn từ MOCK_E1–E3 của bản trước, chỉ đổi ID đơn vị; không gọi là chuyên gia thật. CODAS vẫn là crisp min-max extension, threshold close tau=.02. Cột hằng không có sức phân biệt. Hạng đồng điểm giữ nguyên và UI hiển thị tất cả phương án hạng 1.

## 9. Module và interface mới

- `src/irrigation/preprocess.py`: source → daily, causal boundary, holdout validation; `soil_response.py` chứa constrained ridge.
- `src/irrigation/run_irrigation.py`: orchestration; interfaces, scenario, simulator, KPI và strategies nằm trong các file riêng của src/irrigation.
- `pipelines.run_mode2`: mặc định real; repo này chỉ chạy real; không có fallback hoặc fixture mock.
- `RealStrategyInput`: step, moisture_observation_index, weather_forecast, water_budget_liters, config, fitted model.
- `RealStrategyOutput`: irrigation_liters, used_fallback, reason.
- Model chỉ chứa hệ số và statistics training, không chứa targets hay quan trắc holdout.

Module SIDSS mới của Quyên cần nhận interface real hoặc có adapter đơn vị rõ ràng, không đưa litre vào field irrigation_mm của interface cũ.

## 10. Tính lặp lại và nguồn

Các source CSV được copy nguyên byte từ ZIP và ghi SHA-256. Không chạy code đi kèm dataset, không gọi lại API weather, không tải thêm file và không thay input gốc. `prepare_src/irrigation/preprocess.py` và toàn bộ pipeline chạy offline khi dependencies đã cài. Source README còn placeholder DOI/tác giả; giữ trạng thái đó trong provenance thay vì suy đoán.
