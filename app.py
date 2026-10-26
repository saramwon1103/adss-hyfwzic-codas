"""Local Streamlit UI for DSS coursework."""

from datetime import date

import pandas as pd
import streamlit as st

from src.core.codas import crisp_codas
from src.irrigation.scenario_generator import SCENARIOS
from src.pipelines import run_mode1, run_mode2
from src.visualization.tables import winner_names


# ============================================================
# LABEL HELPERS
# ============================================================

def criterion_label(name):
    """
    Việt hóa tên các tiêu chí KPI dùng trong Mode 2.
    Backend vẫn giữ nguyên tên cột tiếng Anh.
    """

    labels = {
        "water_use_liters":
            "Lượng nước sử dụng",

        "soil_moisture_mae_points":
            "Sai số độ ẩm đất",

        "over_irrigation_liters":
            "Lượng tưới vượt mức",

        "replanning_response_days":
            "Thời gian phản ứng khi điều chỉnh",

        "missing_data_robustness":
            "Khả năng chịu thiếu dữ liệu",
    }

    return labels.get(
        name,
        name,
    )


def vietnamese_dataframe(df):
    """
    Việt hóa tên cột/index chỉ phục vụ hiển thị UI.

    Không thay đổi DataFrame gốc được sử dụng bởi
    Hy-FWZIC, CODAS, simulator hoặc các pipeline.
    """

    result = df.copy()

    column_labels = {
        # ----------------------------------------------------
        # Chung
        # ----------------------------------------------------
        "alternative":
            "Phương án",

        "criterion":
            "Tiêu chí",

        "scenario":
            "Kịch bản",

        "rank":
            "Thứ hạng",

        "weight":
            "Trọng số",

        # ----------------------------------------------------
        # Hy-FWZIC
        # ----------------------------------------------------
        "mu":
            "Độ thuộc μ",

        "nu":
            "Độ không thuộc ν",

        "crisp":
            "Giá trị rõ",

        "expected_weight":
            "Trọng số công bố",

        "abs_error":
            "Sai số tuyệt đối",

        # ----------------------------------------------------
        # CODAS
        # ----------------------------------------------------
        "H":
            "Điểm CODAS H",

        "E":
            "Khoảng cách Euclid",

        "T":
            "Khoảng cách Taxicab",

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------
        "timestamp":
            "Thời gian",

        "sector_id":
            "Khu vực",

        "temperature_c":
            "Nhiệt độ (°C)",

        "humidity_percent":
            "Độ ẩm không khí (%)",

        "rain_mm":
            "Lượng mưa (mm)",

        "radiation_mj_m2":
            "Bức xạ (MJ/m²)",

        "historical_irrigation_liters":
            "Lượng nước tưới lịch sử (L)",

        "irrigation_duration_min":
            "Thời gian tưới (phút)",

        "soil_moisture_index":
            "Chỉ số độ ẩm đất",

        "next_soil_moisture_index":
            "Độ ẩm đất ngày kế tiếp",

        "sensor_missing":
            "Thiếu dữ liệu cảm biến",

        "scenario_sensor_outage":
            "Mất cảm biến do kịch bản",

        "demand_event":
            "Có nhu cầu tưới",

        "water_budget_liters":
            "Giới hạn nước tưới (L)",

        # ----------------------------------------------------
        # Weather forecast
        # ----------------------------------------------------
        "temperature_c_forecast":
            "Dự báo nhiệt độ (°C)",

        "humidity_percent_forecast":
            "Dự báo độ ẩm không khí (%)",

        "rain_mm_forecast":
            "Dự báo lượng mưa (mm)",

        "radiation_mj_m2_forecast":
            "Dự báo bức xạ (MJ/m²)",

        # ----------------------------------------------------
        # Strategy trace
        # ----------------------------------------------------
        "moisture_start_index":
            "Độ ẩm đầu ngày",

        "moisture_observation_index":
            "Độ ẩm quan sát",

        "moisture_end_index":
            "Độ ẩm cuối ngày",

        "irrigation_liters":
            "Lượng nước tưới (L)",

        "modeled_excess_irrigation_liters":
            "Lượng tưới vượt mức (L)",

        "modeled_feasible_demand_liters":
            "Nhu cầu nước ước tính (L)",

        "used_fallback":
            "Có dùng cơ chế dự phòng",

        "outside_training_range":
            "Ngoài miền huấn luyện",

        "environment_disturbance_points":
            "Biến động môi trường",

        "reason":
            "Lý do quyết định",

        # ----------------------------------------------------
        # Validation / residuals
        # ----------------------------------------------------
        "observed_next_index":
            "Độ ẩm thực tế ngày sau",

        "predicted_next_index":
            "Độ ẩm mô hình dự báo",

        "persistence_next_index":
            "Dự báo Persistence",

        "residual":
            "Sai số residual",

        # ----------------------------------------------------
        # Sensitivity
        # ----------------------------------------------------
        "winner":
            "Phương án đứng đầu",

        "top_alternative":
            "Phương án đứng đầu",

        "baseline_rank":
            "Hạng ban đầu",

        "sensitivity_rank":
            "Hạng sau thay đổi",

        "baseline_H":
            "Điểm H ban đầu",

        "sensitivity_H":
            "Điểm H sau thay đổi",

        "rank_change":
            "Thay đổi thứ hạng",

        "baseline_weight":
            "Trọng số ban đầu",

        "sensitivity_weight":
            "Trọng số sau thay đổi",

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------
        "water_use_liters":
            "Lượng nước sử dụng (L)",

        "soil_moisture_mae_points":
            "Sai số độ ẩm đất",

        "over_irrigation_liters":
            "Lượng tưới vượt mức (L)",

        "replanning_response_days":
            "Thời gian phản ứng (ngày)",

        "missing_data_robustness":
            "Khả năng chịu thiếu dữ liệu",
    }

    result = result.rename(
        columns={
            col:
                column_labels.get(
                    col,
                    criterion_label(col),
                )
            for col
            in result.columns
        }
    )

    # Nếu index là tên KPI thì Việt hóa luôn.
    result = result.rename(
        index=lambda value:
            criterion_label(value)
            if isinstance(value, str)
            else value
    )

    return result


# ============================================================
# SENSITIVITY HELPERS
# ============================================================

def reweight_one_criterion(
    weights,
    criterion,
    new_weight,
):
    """
    Thay trọng số của một tiêu chí và phân bổ phần trọng số
    còn lại theo đúng tỷ lệ tương đối ban đầu.

    Tổng trọng số sau cùng luôn bằng 1.
    """

    weights = (
        weights
        .astype(float)
        .copy()
    )

    if criterion not in weights.index:
        raise ValueError(
            f"Không tìm thấy tiêu chí: {criterion}"
        )

    new_weight = float(
        new_weight
    )

    if not 0.0 < new_weight < 1.0:
        raise ValueError(
            "Trọng số mới phải nằm trong khoảng (0, 1)."
        )

    others = (
        weights
        .drop(
            index=criterion
        )
    )

    other_total = float(
        others.sum()
    )

    if other_total <= 0:
        raise ValueError(
            "Không thể phân bổ lại trọng số "
            "cho các tiêu chí còn lại."
        )

    remaining = (
        1.0
        - new_weight
    )

    adjusted = (
        others
        / other_total
        * remaining
    )

    result = (
        weights.copy()
    )

    result.loc[
        criterion
    ] = new_weight

    result.loc[
        adjusted.index
    ] = adjusted

    # tránh sai số floating point nhỏ
    result = (
        result
        / result.sum()
    )

    return result


def ranking_comparison(
    baseline,
    sensitivity,
):
    """
    So sánh thứ hạng CODAS trước và sau khi thay đổi trọng số.
    """

    left = (
        baseline[
            [
                "alternative",
                "H",
                "rank",
            ]
        ]
        .rename(
            columns={
                "H":
                    "Điểm H ban đầu",

                "rank":
                    "Hạng ban đầu",
            }
        )
    )

    right = (
        sensitivity[
            [
                "alternative",
                "H",
                "rank",
            ]
        ]
        .rename(
            columns={
                "H":
                    "Điểm H sau thay đổi",

                "rank":
                    "Hạng sau thay đổi",
            }
        )
    )

    comparison = (
        left.merge(
            right,
            on="alternative",
            how="inner",
        )
    )

    comparison[
        "Thay đổi hạng"
    ] = (
        comparison[
            "Hạng ban đầu"
        ]
        -
        comparison[
            "Hạng sau thay đổi"
        ]
    )

    comparison = (
        comparison.rename(
            columns={
                "alternative":
                    "Phương án",
            }
        )
    )

    return (
        comparison
        .sort_values(
            [
                "Hạng sau thay đổi",
                "Phương án",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hệ hỗ trợ ra quyết định nông nghiệp",
    page_icon="🌱",
    layout="wide",
)

st.title(
    "🌱 Hệ hỗ trợ ra quyết định nông nghiệp"
)

st.caption(
    "Đồ án Hệ hỗ trợ ra quyết định · "
    "Hy-FWZIC + CODAS · Agriculture 4.0"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Chọn chế độ"
)

mode = (
    st.sidebar.radio(
        "Bạn muốn xem phần nào?",
        [
            "Mode 1 · Tái lập kết quả bài báo",
            "Mode 2 · Thí nghiệm tưới thông minh",
        ],
    )
)


# ============================================================
# MODE 1
# ============================================================

if mode.startswith(
    "Mode 1"
):

    st.header(
        "Mode 1 · Tái lập kết quả bài báo"
    )

    st.info(
        "Mục tiêu của Mode 1 là kiểm tra xem nhóm có thể "
        "tái tạo lại trọng số Hy-FWZIC, kết quả CODAS "
        "và phân tích độ nhạy của bài báo hay không."
    )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    source = (
        st.sidebar.selectbox(
            "Nguồn trọng số",
            [
                "published",
                "computed",
            ],
            format_func=lambda value:
                (
                    "Trọng số công bố trong bài báo"
                    if value == "published"
                    else
                    "Trọng số nhóm tính lại bằng Hy-FWZIC"
                ),
        )
    )

    tau = (
        st.sidebar.slider(
            "Ngưỡng τ của CODAS",
            min_value=0.0,
            max_value=0.05,
            value=0.02,
            step=0.001,
        )
    )

    run = run_mode1(
        tau,
        source,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    a, b, c = (
        st.columns(3)
    )

    a.metric(
        "Phương án đứng đầu",
        run[
            "result"
        ]
        .ranking
        .iloc[0]
        .alternative,
    )

    b.metric(
        "Khớp kết quả Bảng 8",
        f"{run['comparison'].score_within_tolerance.sum()}/13",
    )

    c.metric(
        "Khớp kết quả Bảng 10",
        f"{run['sensitivity_comparison'].score_within_tolerance.sum()}/104",
    )

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    tab1, tab2, tab3 = (
        st.tabs(
            [
                "1 · Dữ liệu & trọng số",
                "2 · Kết quả xếp hạng",
                "3 · Phân tích độ nhạy",
            ]
        )
    )

    # --------------------------------------------------------
    # Tab 1
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "Ma trận đánh giá các ADSS"
        )

        st.dataframe(
            vietnamese_dataframe(
                run[
                    "matrix"
                ]
            ),
            width="stretch",
        )

        st.subheader(
            "Trọng số tiêu chí"
        )

        st.bar_chart(
            run[
                "weights"
            ]
        )

        st.subheader(
            "So sánh trọng số tính lại với bài báo"
        )

        st.dataframe(
            vietnamese_dataframe(
                run[
                    "weight_comparison"
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    # --------------------------------------------------------
    # Tab 2
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "Kết quả xếp hạng CODAS"
        )

        st.dataframe(
            vietnamese_dataframe(
                run[
                    "result"
                ].ranking
            ),
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            run[
                "result"
            ]
            .ranking
            .set_index(
                "alternative"
            )
            .H
        )

    # --------------------------------------------------------
    # Tab 3
    # --------------------------------------------------------

    with tab3:

        st.write(
            "Phân tích độ nhạy kiểm tra xem thứ hạng "
            "có thay đổi khi trọng số tiêu chí thay đổi hay không."
        )

        choice = (
            st.selectbox(
                "Chọn kịch bản thay đổi trọng số",
                run[
                    "sensitivity_summary"
                ].scenario,
            )
        )

        st.subheader(
            "Tổng hợp các kịch bản"
        )

        st.dataframe(
            vietnamese_dataframe(
                run[
                    "sensitivity_summary"
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        st.subheader(
            "Chi tiết kịch bản đang chọn"
        )

        st.dataframe(
            vietnamese_dataframe(
                run[
                    "sensitivity"
                ].query(
                    "scenario == @choice"
                )
            ),
            hide_index=True,
            width="stretch",
        )

        with st.expander(
            "Xem đối chiếu reproduction chi tiết"
        ):

            st.dataframe(
                vietnamese_dataframe(
                    run[
                        "comparison"
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

            st.dataframe(
                vietnamese_dataframe(
                    run[
                        "sensitivity_comparison"
                    ]
                ),
                hide_index=True,
                width="stretch",
            )


# ============================================================
# MODE 2
# ============================================================

else:

    st.header(
        "Mode 2 · Thí nghiệm tưới thông minh"
    )

    st.info(
        "Mode 2 là phần mở rộng của nhóm: "
        "sử dụng dữ liệu tưới thực tế để so sánh "
        "SIDSS-like, FDSS-like và MRGCD-like."
    )

    # --------------------------------------------------------
    # Sidebar configuration
    # --------------------------------------------------------

    st.sidebar.header(
        "Thiết lập thí nghiệm"
    )

    scenario = (
        st.sidebar.selectbox(
            "Kịch bản môi trường",
            SCENARIOS,
            format_func=lambda value:
                {
                    "Normal":
                        "Bình thường",

                    "Drought":
                        "Hạn hán",

                    "Missing Sensor":
                        "Mất dữ liệu cảm biến",
                }.get(
                    value,
                    value,
                ),
        )
    )

    sector = (
        st.sidebar.selectbox(
            "Khu vực cà chua",
            [
                1,
                2,
            ],
        )
    )

    period = (
        st.sidebar.date_input(
            "Khoảng thời gian đánh giá",
            (
                date(
                    2025,
                    6,
                    1,
                ),
                date(
                    2025,
                    7,
                    13,
                ),
            ),
            min_value=date(
                2025,
                4,
                1,
            ),
            max_value=date(
                2025,
                9,
                24,
            ),
        )
    )

    if len(period) != 2:

        st.warning(
            "Vui lòng chọn đủ ngày bắt đầu "
            "và ngày kết thúc."
        )

        st.stop()

    # --------------------------------------------------------
    # Execute pipeline
    # --------------------------------------------------------

    try:

        run = (
            run_mode2(
                scenario,
                sector=sector,
                start=(
                    period[
                        0
                    ].isoformat()
                ),
                end=(
                    period[
                        1
                    ].isoformat()
                ),
            )
        )

    except (
        ValueError,
        FileNotFoundError,
    ) as error:

        st.error(
            str(
                error
            )
        )

        st.stop()

    q = (
        run[
            "quality"
        ]
    )

    # --------------------------------------------------------
    # Research integrity
    # --------------------------------------------------------

    with st.expander(
        "Lưu ý về phạm vi nghiên cứu"
    ):

        st.write(
            "• Dữ liệu đầu vào là dữ liệu cảm biến/van tưới "
            "thực tế kết hợp dữ liệu thời tiết tái phân tích."
        )

        st.write(
            "• SIDSS-like, FDSS-like và MRGCD-like là các "
            "chiến lược proxy được nhóm tái dựng dựa trên "
            "mô tả của các hệ thống gốc."
        )

        st.write(
            "• Đây không phải mã nguồn chính thức "
            "của SIDSS, FDSS hoặc MRGCD."
        )

        st.write(
            "• Kết quả là kết quả mô phỏng phản thực "
            "(counterfactual/modelled outcomes), "
            "chưa phải kết quả kiểm chứng ngoài đồng ruộng."
        )

        st.write(
            "• Trọng số chuyên gia hiện tại mang tính minh họa."
        )

    # --------------------------------------------------------
    # Top summary
    # --------------------------------------------------------

    winners = (
        winner_names(
            run[
                "result"
            ]
        )
    )

    a, b, c, d = (
        st.columns(4)
    )

    a.metric(
        "Kịch bản",
        {
            "Normal":
                "Bình thường",

            "Drought":
                "Hạn hán",

            "Missing Sensor":
                "Mất cảm biến",
        }.get(
            scenario,
            scenario,
        ),
    )

    b.metric(
        "Phương án đứng đầu",
        ", ".join(
            winners
        ),
    )

    c.metric(
        "Số ngày đánh giá",
        len(
            run[
                "dataset"
            ]
        ),
    )

    d.metric(
        "Khu vực",
        f"Sector {sector}",
    )

    # ========================================================
    # MAIN TABS
    # ========================================================

    tab1, tab2, tab3 = (
        st.tabs(
            [
                "1 · Tổng quan thí nghiệm",
                "2 · Kết quả & xếp hạng",
                "3 · Phân tích độ nhạy",
            ]
        )
    )

    # ========================================================
    # TAB 1 · OVERVIEW
    # ========================================================

    with tab1:

        st.subheader(
            "Flow của một lần chạy"
        )

        st.markdown(
            """
**Bước 1 · Chọn dữ liệu và kịch bản**

Dữ liệu cảm biến, van tưới và thời tiết được lấy theo
sector và khoảng thời gian đã chọn.

↓

**Bước 2 · Tiền xử lý dữ liệu**

Dữ liệu được tổng hợp theo ngày và tạo các biến cần thiết
cho thí nghiệm.

↓

**Bước 3 · Xây dựng mô hình phản ứng độ ẩm đất**

Mô hình surrogate được học chỉ từ dữ liệu trước giai đoạn
đánh giá.

↓

**Bước 4 · Tạo biến động môi trường**

Sai số mô hình trên dữ liệu huấn luyện được replay để
môi trường mô phỏng không trở thành một vòng lặp lý tưởng.

↓

**Bước 5 · Chạy ba chiến lược tưới**

- SIDSS-like
- FDSS-like
- MRGCD-like

↓

**Bước 6 · Tính KPI**

Kết quả của mỗi chiến lược được chuyển thành ma trận KPI.

↓

**Bước 7 · Hy-FWZIC**

Hy-FWZIC xác định mức độ quan trọng của các tiêu chí.

↓

**Bước 8 · CODAS**

CODAS kết hợp KPI và trọng số để xếp hạng các phương án.

↓

**Bước 9 · Phân tích độ nhạy**

Thay đổi trọng số tiêu chí và chạy CODAS lại để kiểm tra
độ ổn định của kết quả.
"""
        )

        st.divider()

        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        st.subheader(
            "Thông tin thí nghiệm"
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        c1.metric(
            "Số ngày đánh giá",
            len(
                run[
                    "dataset"
                ]
            ),
        )

        c2.metric(
            "Residual huấn luyện",
            len(
                run[
                    "training_residuals"
                ]
            ),
        )

        c3.metric(
            "Disturbance replay",
            len(
                run[
                    "disturbances"
                ]
            ),
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        with st.expander(
            "Xem dữ liệu đầu vào"
        ):

            st.dataframe(
                vietnamese_dataframe(
                    run[
                        "dataset"
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

        # ----------------------------------------------------
        # Training residuals
        # ----------------------------------------------------

        with st.expander(
            "Xem residual dùng để tạo biến động môi trường"
        ):

            st.dataframe(
                vietnamese_dataframe(
                    run[
                        "training_residuals"
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

        # ----------------------------------------------------
        # Strategy trace
        # ----------------------------------------------------

        with st.expander(
            "Xem chi tiết mô phỏng từng chiến lược"
        ):

            strategy = (
                st.selectbox(
                    "Chọn chiến lược",
                    list(
                        run[
                            "traces"
                        ]
                    ),
                    key=(
                        "trace_strategy"
                    ),
                )
            )

            st.dataframe(
                vietnamese_dataframe(
                    run[
                        "traces"
                    ][
                        strategy
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    # ========================================================
    # TAB 2 · RESULT
    # ========================================================

    with tab2:

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        st.subheader(
            "Bước 1 · Ma trận KPI của ba phương án"
        )

        st.write(
            "Ma trận này là đầu vào cho CODAS. "
            "Mỗi hàng là một chiến lược tưới; "
            "mỗi cột là một tiêu chí đánh giá."
        )

        st.dataframe(
            vietnamese_dataframe(
                run[
                    "matrix"
                ]
            ),
            width="stretch",
        )

        constants = (
            run[
                "matrix"
            ]
            .columns[
                (
                    run[
                        "matrix"
                    ].max()
                    -
                    run[
                        "matrix"
                    ].min()
                )
                < 1e-10
            ]
            .tolist()
        )

        if constants:

            st.info(
                "Trong lần chạy này, các KPI sau "
                "không giúp phân biệt ba phương án: "
                +
                ", ".join(
                    criterion_label(
                        col
                    )
                    for col
                    in constants
                )
            )

        # ----------------------------------------------------
        # Weights
        # ----------------------------------------------------

        st.subheader(
            "Bước 2 · Trọng số tiêu chí từ Hy-FWZIC"
        )

        weights_display = (
            run[
                "weights"
            ]
            .rename(
                index=criterion_label
            )
        )

        st.bar_chart(
            weights_display
        )

        st.dataframe(
            weights_display
            .rename(
                "Trọng số"
            )
            .to_frame(),
            width="stretch",
        )

        # ----------------------------------------------------
        # CODAS
        # ----------------------------------------------------

        st.subheader(
            "Bước 3 · Xếp hạng bằng CODAS"
        )

        ranking = (
            run[
                "result"
            ].ranking
        )

        st.dataframe(
            vietnamese_dataframe(
                ranking
            ),
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            ranking
            .set_index(
                "alternative"
            )
            .H
        )

        # ----------------------------------------------------
        # Automatic interpretation
        # ----------------------------------------------------

        st.subheader(
            "Cách đọc kết quả"
        )

        if len(
            winners
        ) == 1:

            st.success(
                f"Trong kịch bản này, {winners[0]} "
                "đang có thứ hạng CODAS cao nhất."
            )

        else:

            st.info(
                "Có đồng hạng ở vị trí thứ nhất: "
                +
                ", ".join(
                    winners
                )
                +
                ". Đây là kết quả hợp lệ, không cần "
                "cố phá tie."
            )

        if not q[
            "model_beats_persistence_mae"
        ]:

            st.warning(
                "Mô hình surrogate hiện chưa vượt rõ "
                "baseline persistence trên holdout. "
                "Do đó kết quả xếp hạng chỉ nên được "
                "xem là kết quả thăm dò / proof-of-concept."
            )

        # ----------------------------------------------------
        # Optional trajectory
        # ----------------------------------------------------

        with st.expander(
            "Xem biểu đồ độ ẩm và lượng nước tưới"
        ):

            traces = (
                pd.concat(
                    run[
                        "traces"
                    ].values(),
                    ignore_index=True,
                )
            )

            st.markdown(
                "**Độ ẩm đất mô phỏng**"
            )

            st.line_chart(
                traces.pivot(
                    index="timestamp",
                    columns="alternative",
                    values="moisture_end_index",
                )
            )

            st.markdown(
                "**Lượng nước tưới theo ngày**"
            )

            st.bar_chart(
                traces.pivot(
                    index="timestamp",
                    columns="alternative",
                    values="irrigation_liters",
                )
            )

    # ========================================================
    # TAB 3 · SENSITIVITY
    # ========================================================

    with tab3:

        st.subheader(
            "Nếu thay đổi mức độ quan trọng của tiêu chí thì sao?"
        )

        st.write(
            "Ở bước này, KPI của kịch bản được giữ nguyên. "
            "Ta chỉ thay đổi trọng số của một tiêu chí, "
            "phân bổ lại phần trọng số còn lại, "
            "sau đó chạy CODAS lại."
        )

        st.info(
            "Đây là phân tích độ nhạy của trọng số. "
            "Không phải thu thập lại ý kiến chuyên gia "
            "và không phải chạy Hy-FWZIC từ đầu."
        )

        baseline_weights = (
            run[
                "weights"
            ]
            .astype(float)
            .copy()
        )

        selected_criterion = (
            st.selectbox(
                "Chọn tiêu chí muốn thay đổi",
                list(
                    baseline_weights.index
                ),
                format_func=criterion_label,
                key=(
                    "sensitivity_criterion"
                ),
            )
        )

        baseline_weight = float(
            baseline_weights.loc[
                selected_criterion
            ]
        )

        new_weight = (
            st.slider(
                "Trọng số mới",
                min_value=0.05,
                max_value=0.60,
                value=float(
                    min(
                        max(
                            baseline_weight,
                            0.05,
                        ),
                        0.60,
                    )
                ),
                step=0.01,
                key=(
                    "sensitivity_weight"
                ),
            )
        )

        # ----------------------------------------------------
        # Reweight
        # ----------------------------------------------------

        sensitivity_weights = (
            reweight_one_criterion(
                baseline_weights,
                selected_criterion,
                new_weight,
            )
        )

        # ----------------------------------------------------
        # CODAS rerun
        # ----------------------------------------------------

        sensitivity_result = (
            crisp_codas(
                run[
                    "matrix"
                ],
                sensitivity_weights,
                run[
                    "criteria"
                ].direction,
            )
        )

        comparison = (
            ranking_comparison(
                run[
                    "result"
                ].ranking,
                sensitivity_result.ranking,
            )
        )

        baseline_winners = (
            run[
                "result"
            ]
            .ranking
            .loc[
                run[
                    "result"
                ]
                .ranking[
                    "rank"
                ]
                == 1,
                "alternative",
            ]
            .tolist()
        )

        sensitivity_winners = (
            sensitivity_result
            .ranking
            .loc[
                sensitivity_result
                .ranking[
                    "rank"
                ]
                == 1,
                "alternative",
            ]
            .tolist()
        )

        winner_changed = (
            set(
                baseline_winners
            )
            !=
            set(
                sensitivity_winners
            )
        )

        # ----------------------------------------------------
        # Flow
        # ----------------------------------------------------

        st.subheader(
            "Flow phân tích độ nhạy"
        )

        st.code(
            (
                f"Kịch bản: {scenario}\n"
                "\n"
                "Ma trận KPI hiện tại\n"
                "        ↓\n"
                "Trọng số Hy-FWZIC ban đầu\n"
                "        ↓\n"
                "CODAS ban đầu\n"
                "        ↓\n"
                f"Thay đổi tiêu chí: "
                f"{criterion_label(selected_criterion)}\n"
                f"{baseline_weight:.3f} "
                f"→ {new_weight:.3f}\n"
                "        ↓\n"
                "Phân bổ lại trọng số còn lại\n"
                "        ↓\n"
                "Chạy CODAS lại\n"
                "        ↓\n"
                "So sánh thứ hạng"
            ),
            language=None,
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        s1, s2, s3 = (
            st.columns(3)
        )

        s1.metric(
            "Trọng số ban đầu",
            f"{baseline_weight:.3f}",
        )

        s2.metric(
            "Trọng số mới",
            f"{new_weight:.3f}",
        )

        s3.metric(
            "Phương án đứng đầu có đổi không?",
            (
                "Có"
                if winner_changed
                else
                "Không"
            ),
        )

        # ----------------------------------------------------
        # Weight comparison
        # ----------------------------------------------------

        with st.expander(
            "Xem trọng số trước và sau thay đổi"
        ):

            compare_weights = (
                pd.DataFrame(
                    {
                        "Trọng số ban đầu":
                            baseline_weights,

                        "Trọng số mới":
                            sensitivity_weights,
                    }
                )
            )

            compare_weights.index = [
                criterion_label(
                    name
                )
                for name
                in compare_weights.index
            ]

            st.dataframe(
                compare_weights,
                width="stretch",
            )

        # ----------------------------------------------------
        # Rank comparison
        # ----------------------------------------------------

        st.subheader(
            "So sánh thứ hạng trước và sau"
        )

        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
        )

        # ----------------------------------------------------
        # New ranking
        # ----------------------------------------------------

        st.subheader(
            "Kết quả CODAS sau khi thay đổi trọng số"
        )

        st.dataframe(
            vietnamese_dataframe(
                sensitivity_result.ranking
            ),
            hide_index=True,
            width="stretch",
        )

        st.bar_chart(
            sensitivity_result
            .ranking
            .set_index(
                "alternative"
            )
            .H
        )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        if winner_changed:

            st.warning(
                "Có thay đổi ở tập phương án đứng đầu. "
                "Kết quả lựa chọn nhạy với thay đổi "
                "trọng số này."
            )

        else:

            st.success(
                "Phương án đứng đầu không thay đổi "
                "ở mức trọng số hiện tại."
            )

        # ----------------------------------------------------
        # Predefined scenarios
        # ----------------------------------------------------

        with st.expander(
            "Xem các kịch bản độ nhạy tự động của hệ thống"
        ):

            st.dataframe(
                vietnamese_dataframe(
                    run[
                        "sensitivity_summary"
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

            st.dataframe(
                vietnamese_dataframe(
                    run[
                        "sensitivity"
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "Chi tiết kỹ thuật & kiểm định mô hình"
    ):

        v1, v2, v3 = (
            st.columns(3)
        )

        v1.metric(
            "MAE mô hình",
            f'{q["holdout_mae_points"]:.3f}',
        )

        v2.metric(
            "MAE Persistence",
            f'{q["persistence_mae_points"]:.3f}',
        )

        v3.metric(
            "Số cặp holdout hợp lệ",
            q[
                "holdout_pairs"
            ],
        )

        st.caption(
            "Phần này dành cho kiểm tra kỹ thuật. "
            "Future holdout target không được dùng "
            "trong training hoặc policy input."
        )

        st.subheader(
            "Bảng kiểm định holdout"
        )

        st.dataframe(
            vietnamese_dataframe(
                run[
                    "validation"
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        st.subheader(
            "Thông tin chất lượng dữ liệu"
        )

        st.json(
            q
        )

        st.subheader(
            "Thông tin mô hình surrogate"
        )

        st.json(
            run[
                "model"
            ].to_dict()
        )