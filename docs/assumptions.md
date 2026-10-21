# Giả định hiện hành v1.1

**Mode 2 mặc định là dữ liệu thực.** Repo này không chứa chế độ mock. Không gán root_depth=300 mm, target VWC=.28 hoặc cap=12 mm/ngày của bản minh họa cũ vào dữ liệu thực.

## Mode 1 giữ nguyên

- Score HyFS `(2mu−mu*nu+1)/(2nu+1)`; scalar `(1−(1−mu)^lambda,nu^lambda)`.
- Scale mỗi expert bằng nghịch đảo tổng score HyFS, aggregate 5 expert đều nhau, defuzzify và normalize.
- Cặp C7 ở Table 5 có nu in không nhất quán; reference tính lại nu≈.86304557, không dùng lỗi in để tính ngược weight.
- Hyperbolic E là tổng khoảng cách từng cặp criterion, T là tổng Hamming từng cặp, NIS=(min mu,max nu).
- h=(Ei−Ek)+psi(Ei−Ek)*(Ti−Tk), H=sum h.
- Paper Eq.16 dùng close rule `abs(Ei−Ek)<=tau`; tau=.02 là giả định tái lập công khai vì bài không chỉ rõ một giá trị số.
- H in 3 chữ số thường bị cắt phần thập phân: tolerance .0011; E/T tolerance 5.1e-7.
- Giữ đồng hạng, competition/min rank. S1..S8: một weight=.3, các weight khác=.1.

## Mode 2 thực

Chi tiết đầy đủ trong `REAL_DATA_UPDATE.md`. Các giả định vận hành chính:

1. Chỉ dùng sector 1/2 cà chua ngoài đồng.
2. Độ ẩm là probe-index points, water là nominal-flow L. Không có area/calibrated VWC/ET0 nên không tự suy thành mm hoặc m³/m³.
3. Sensor đầu ngày: backward asof tuổi≤12h. Không lấp từ tương lai; range check 0–100, chưa khẳng định đã xử lý mọi spike/flatline.
4. Dự báo thời tiết là persistence ngày trước; weather thực ngày này chỉ dùng trong surrogate environment.
5. Model ridge alpha10, constraint AR∈[0,1], water gain≥0; scaler, coefficients và cap chỉ tính trên training targets trước evaluation.
6. Target50 midpoint MADP40/FC60, không phải độ ẩm mục tiêu thực nghiệm do tác giả kiểm định cho mọi trường hợp.
7. Cap là percentile95 của daily nominal volume trên valid training transitions, không phải capacity thực của toàn sector.
8. Counterfactual model không bảo đảm physical water balance; các KPI excess/response là proxy.
9. Heuristic FDSS/MRGCD và SIDSS stub; không có ANFIS/PLSR/IDSCU gốc. Expert weights vẫn mẫu.
10. Drought/added outage là what-if trên dữ liệu nguồn. Kết quả mode này có input thực nhưng outcome policy do mô hình tạo.
11. Đánh giá surrogate và baseline được xuất cả khi surrogate kém baseline. Ngoại suy và đáp ứng nước yếu được nêu rõ; không tạo hạng duy nhất nếu score đồng hạng.

## CODAS cho KPI

Crisp min-max: benefit=(x−min)/(max−min), cost=(max−x)/(max−min), cột hằng normalized=1 cho mọi alternative. Weighted=normalized×weight; NIS=min; E=L2, T=L1. Close rule tau=.02. Đây là biến thể minh họa khác với fuzzy CODAS Mode 1. Sensitivity Mode 2: một KPI=.4, còn lại=.15.
