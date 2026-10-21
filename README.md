# adss-hyfwzic-codas

Bản chạy được theo cấu trúc `src/`, `data/paper/`, `data/irrigation/` của ZIP khung người dùng gửi. Repo đích: https://github.com/saramwon1103/adss-hyfwzic-codas.

**Mode 1** tái lập Hy-FWZIC/CODAS và sensitivity của paper. **Mode 2** mặc định dùng sensor/actuator/weather upload Arnesano, sector cà chua ngoài đồng 1/2; không có fallback về mock.

Kết quả Mode 2 là counterfactual simulation bằng constrained-ridge surrogate, không phải hiệu quả đo ngoài đồng của ba policy. SIDSS vẫn là stub; FDSS/MRGCD là heuristic; preferences vẫn minh họa. Đơn vị L và relative probe-index points, không gán thành mm/VWC khi không có area/hiệu chuẩn.

## Chạy nhanh trên Linux

Mở terminal tại thư mục gốc repo (có app.py và src), dùng Python 3.12:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest
python -m src.paper.reproduce_paper
python -m src.irrigation.run_irrigation --scenario all
python -m streamlit run app.py
```

Mở http://localhost:8501. Root wrappers vẫn dùng được:

```bash
python reproduce_paper.py
python run_irrigation.py --scenario all
python prepare_real_data.py --sector 1
```

Lựa chọn khác:

```bash
python -m src.paper.reproduce_paper --weights computed --output outputs/paper_computed
python -m src.irrigation.run_irrigation --sector 2 --scenario all --output outputs/irrigation_sector2
python -m src.irrigation.run_irrigation --start 2025-06-01 --end 2025-07-13 --scenario Drought
python -m src.irrigation.preprocess --sector 1
python make_figures.py
```

## Cấu trúc và trách nhiệm

| Đường dẫn | Nội dung |
|---|---|
| src/core/fuzzy.py, hy_fwzic.py, codas.py | Fuzzy, weighting và hai biến thể CODAS |
| src/paper/load_paper_data.py | Nạp dữ liệu paper và mapping |
| src/paper/reproduce_paper.py | Pipeline Mode 1 và CLI |
| src/paper/validate_paper.py | So expected weights/H/ranks, không thay reference để ép khớp |
| src/analysis/sensitivity.py, stability.py | Weight scenarios, rank movement, tie-aware correlation |
| src/irrigation/preprocess.py | Source 10 phút → daily, causal sensor boundary, holdout split |
| src/irrigation/soil_response.py | Model fit/scaler/cap training-only |
| src/irrigation/interfaces.py | StrategyInput/Output theo L/index |
| src/irrigation/scenario_generator.py | Đọc cấu hình scenarios.csv và tạo what-if forcing |
| src/irrigation/strategies/ | SIDSS stub, FDSS-like và MRGCD-like |
| src/irrigation/simulator.py | Closed-loop surrogate cho từng policy |
| src/irrigation/kpi_evaluator.py | Năm KPI theo L/index và paired missing robustness |
| src/irrigation/build_performance_matrix.py | Validate/build matrix alternative×KPI |
| src/irrigation/run_irrigation.py | Pipeline Mode 2 và CLI |
| src/visualization/ | Biểu đồ và winner table helpers |
| data/paper/expected/ | Expected outputs Tables 5/7/8/10 |
| data/irrigation/raw/ | Hai CSV gốc 10 phút, provenance và docs nguồn |
| data/irrigation/processed/ | Daily table và quality report |
| outputs/paper/, outputs/irrigation/ | Kết quả sinh lại, được .gitignore bỏ qua mặc định |
| tools/apply_update.py | Preview/apply payload vào Git clone sạch, có backup ngoài repo |

`candidates.csv` là danh mục policy minh họa; registry thực thi trong strategies/__init__.py. `scenarios.csv` là cấu hình được engine đọc. Thay tên/tạo policy mới cần code tương ứng; không chỉ thêm tên vào CSV.

## Kiểm chứng

- **55 tests passed** cho cấu trúc repo mới, gồm UI, Mode1, real-data pipeline và helper cập nhật repo.
- Mode1: 13/13 H gốc; 104/104 H và ranks sensitivity; tolerance H=.0011.
- C5≈.142287487, A8 H≈3.843744, A11 H≈−3.519666; S1→A9, S6→A5.
- Mode2: 43 ngày mặc định, 01/06–13/07/2025. Sector1 train87/holdout39 cặp; model MAE7.4818, persistence7.5897. Sector2 train85/holdout27; model6.8668 kém persistence5.5556, được giữ công khai.
- Có ngoại suy mùa hè và đáp ứng nước yếu: ranking là exploratory. Normal/Missing Sensor có thể đồng kết quả, KPI hằng và đồng hạng; không tạo noise hoặc ép winner.

Bộ test fixture mock cũ không mang vào repo này nên số lượng khác bản 75 test có cả legacy mock. Các kiểm chứng paper và dữ liệu thực được giữ; thêm kiểm thử cập nhật clone và scenario CSV.

## Cập nhật GitHub

Xem `docs/GITHUB_UPDATE.md`. Gói được ghép từ ZIP khung người dùng và implementation đã kiểm thử; **không xác minh được current remote GitHub content từ phiên này vì xác thực**, không có push thực tế. Preview diff trên clone của bạn trước khi áp dụng/commit. Không chép thư mục .git hoặc khởi tạo lịch sử mới đè lên repo hiện có.

## Nguồn

Paper: Alamoodi et al. (2024), Computers and Electronics in Agriculture 227, 109618. https://doi.org/10.1016/j.compag.2024.109618.

Dataset upload: Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation Field Trial (Arnesano, Apulia, Italy, 2025). Source README ghi CC BY4.0, tác giả/DOI còn placeholder; giữ nguyên trạng thái đó, không tạo nguồn giả. Dữ liệu weather nguồn là historical reanalysis, không phải forecast lưu trữ. Xem data/irrigation/raw/SOURCE_README.md và docs/methodology.md.
