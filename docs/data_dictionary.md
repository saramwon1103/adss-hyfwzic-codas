# Data dictionary v1.1

## Mode 1

Các file trong `data/paper/` giữ nguyên. `provenance.json` ánh xạ CSV tới Tables 2/4/5/6/7/8/9/10 của paper. Mode 1 và reference weights không lấy dữ liệu từ bộ Arnesano.

## Mode 2: source

`data/irrigation/raw/sector_1_10min.csv`, `sector_2_10min.csv` được copy nguyên byte từ `02_processed_data/merged/dataset_zone_{1,2}.csv` trong ZIP người dùng. Schema gốc trong SOURCE_DATA_DICTIONARY.md. Provenance có đường dẫn member, byte size và SHA-256; không ghi DOI/tác giả chưa xác định.

## Daily table `data/irrigation/processed/processed_irrigation.csv`

| Cột | Đơn vị/kiểu | Vai trò |
|---|---|---|
| timestamp | ngày local, timezone-naive | Giữ Europe/Rome wall-clock date từ nguồn |
| sector_id | string | ARNESANO_TOMATO_1 hoặc 2 |
| temperature_c | °C | Trung bình weather_temp ngày |
| humidity_percent | % RH | Trung bình weather_humidity ngày |
| rain_mm | mm/day xấp xỉ | Tích phân hourly-rain proxy trên lưới 10 phút, sum/6 |
| radiation_mj_m2 | MJ/m²/day | Tích phân direct radiation, sum×600/10⁶ |
| historical_irrigation_liters | L/day danh định | Tổng liters_total; dùng fit/validation, không đưa vào policy |
| irrigation_duration_minutes | phút/day | Tổng phút van mở của source |
| soil_moisture_index | điểm probe 0–100 hoặc NaN | Quan trắc cuối tại/trước đầu ngày, tuổi≤12h |
| sensor_observed_at | timestamp hoặc NaT | Timestamp nguồn của giá trị boundary |
| sensor_age_hours | giờ hoặc NaN | Độ cũ tại đầu ngày |
| next_soil_moisture_index | điểm probe hoặc NaN | Label đầu ngày kế tiếp; chỉ dùng fit/validation, không phải feature |
| historical_daily_median_index | điểm probe hoặc NaN | Thống kê mô tả, không phải policy input |
| sensor_valid_count | số ô 10 phút | Số giá trị 0–100 không thiếu trong ngày |
| interval_count | số ô | Giữ ngày có 144 ô weather |
| sensor_valid_fraction | 0..1 | valid_count/interval_count, không nội suy cho đẹp độ phủ |
| sensor_missing | bool | Không có boundary sensor hợp lệ tuổi≤12h |
| temperature_c_forecast | °C | Temperature của ngày trước |
| humidity_percent_forecast | % RH | RH của ngày trước |
| rain_mm_forecast | mm/day | Rain proxy của ngày trước |
| radiation_mj_m2_forecast | MJ/m²/day | Radiation của ngày trước |
| is_synthetic | bool | False cho source upload; không có nghĩa outcome strategy là đo thực |
| source_kind | string | Uploaded field sensor/actuator plus reanalysis weather |

Các cột thêm trong evaluation/scenario: water_budget_liters (training percentile95), demand_event (nhãn evaluator), scenario_sensor_outage (outage thêm), irrigation_liters (bản sao lịch sử chỉ để audit), forcing đã biến đổi theo scenario. Feature/label không được phân biệt chỉ bằng việc chúng cùng tồn tại trong CSV: model có allowlist và strategy nhận dataclass riêng.

## Real strategy trace

| Cột | Ý nghĩa |
|---|---|
| moisture_start_index, moisture_end_index | Trạng thái surrogate đầu/cuối ngày, không phải đo thực của policy mới |
| moisture_observation_index | Trạng thái surrogate có thể quan trắc; NaN khi lịch sensor thiếu |
| irrigation_liters | Action đề xuất |
| water_budget_liters | Daily allocation cap |
| modeled_excess_irrigation_liters | Proxy phần action vượt upper-index=60 theo surrogate |
| modeled_feasible_demand_liters | Model-inverted dose để đạt target50, chặn ở cap |
| used_fallback, sensor_missing | Fallback và availability thực tế truyền trong mô phỏng |
| demand_event | Chỉ dùng đo response |
| outside_training_range | Có ít nhất một feature ngoài min/max training |
| reason | Luật hoặc fallback đã dùng |

## KPI và weights

`criteria.csv`: water_use_liters (cost,L), soil_moisture_mae_points (cost,index points), over_irrigation_liters (cost,L proxy), replanning_response_days (cost,day), missing_data_robustness (benefit,0..1).

Công thức trong REAL_DATA_UPDATE.md. `preferences.csv` vẫn MOCK_E1–E3; không phải khảo sát chuyên gia từ dataset.

## Outputs để audit

- `data_quality.json`: coverage, missing/outlier counts, nguồn/đơn vị, split, holdout MAE/RMSE, baseline, ngoại suy, cột KPI không phân biệt.
- `soil_response_model.json`: feature names, scaler training, coefficients, alpha, training range và giới hạn nhân quả.
- `historical_holdout_validation.csv`: sensor đầu/ngày sau, volume lịch sử, prediction và persistence baseline; chỉ các cặp có target thực hợp lệ.
- `run_manifest.json`: input hashes, parameters, runtime/dependency versions.
- CODAS files: ranking, normalized/weighted matrix, NIS, relative matrix, sensitivity, stability và weights.
