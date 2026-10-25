from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'


def test_ui_mode1_published_and_computed():
    app=AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value=='A8'
    assert app.metric[1].value=='13/13'
    app.sidebar.selectbox[0].select('computed').run(timeout=30)
    assert not app.exception
    assert app.metric[2].value=='104/104'


def test_ui_mode2_all_scenarios():
    app=AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.radio[0].set_value('Mode 2 · Irrigation simulation').run(timeout=30)
    assert not app.exception
    assert 'REAL INPUT DATA' in app.info[0].value
    for scenario in ['Drought','Missing Sensor','Normal']:
        app.sidebar.selectbox[0].select(scenario).run(timeout=30)
        assert not app.exception
        assert app.metric[1].value=='43'


def test_ui_second_real_sector_shows_negative_validation():
    app=AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.radio[0].set_value('Mode 2 · Irrigation simulation').run(timeout=30)
    app.sidebar.selectbox[1].select(2).run(timeout=30)
    assert not app.exception
    assert app.metric[2].value=='ARNESANO_TOMATO_2'
    assert any('does not beat simple persistence' in w.value for w in app.warning)

def test_ui_mode2_experiment_flow_and_interactive_sensitivity():
    app = AppTest.from_file(str(APP)).run(timeout=30)

    app.sidebar.radio[0].set_value(
        "Mode 2 · Irrigation simulation"
    ).run(timeout=30)

    assert not app.exception

    # Mode 2 headline metrics are present.
    assert app.metric[0].label == "Top alternative(s)"
    assert app.metric[1].label == "Daily steps"
    assert app.metric[1].value == "43"

    # Interactive sensitivity controls should exist.
    assert len(app.selectbox) >= 3
    assert len(app.slider) >= 1

    # Last slider belongs to Mode 2 sensitivity lab.
    sensitivity_slider = app.slider[-1]

    sensitivity_slider.set_value(0.30).run(timeout=30)

    assert not app.exception

    # The sensitivity metrics should now be rendered.
    labels = [metric.label for metric in app.metric]

    assert "Baseline weight" in labels
    assert "New weight" in labels
    assert "Winner changed?" in labels

def test_ui_mode2_sensitivity_can_change_criterion():
    app = AppTest.from_file(str(APP)).run(timeout=30)

    app.sidebar.radio[0].set_value(
        "Mode 2 · Irrigation simulation"
    ).run(timeout=30)

    assert not app.exception

    # In Mode 2:
    # sidebar selectboxes:
    #   0 = Operational scenario
    #   1 = Tomato sector
    #
    # main selectboxes include:
    #   strategy trace
    #   sensitivity criterion
    sensitivity_selectboxes = [
        box
        for box in app.selectbox
        if box.label == "Criterion to stress"
    ]

    assert len(sensitivity_selectboxes) == 1

    box = sensitivity_selectboxes[0]

    options = list(box.options)

    assert len(options) > 1

    box.select(options[-1]).run(timeout=30)

    assert not app.exception