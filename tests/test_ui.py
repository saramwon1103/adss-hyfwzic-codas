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
