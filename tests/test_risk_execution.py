import pytest

from trading_school_ai.config.settings import RiskConfig, Settings
from trading_school_ai.execution.base import LiveLockError, round_to_step
from trading_school_ai.execution.okx import OKXSpotVenue, check_live_locks, to_okx_symbol
from trading_school_ai.execution.paper import PaperVenue
from trading_school_ai.risk.engine import OrderIntent, RiskEngine, RiskState


def engine(**kw):
    cfg = RiskConfig(cooldown_seconds=0, **kw)
    return RiskEngine(cfg, RiskState(equity=10_000.0))


def intent(qty=0.01, px=30_000.0, side="buy"):
    return OrderIntent("BTCUSDT", side, qty, px)


def test_paper_is_default():
    assert Settings().execution.mode == "paper"
    assert Settings().execution.live_enabled is False


def test_risk_approves_valid_order():
    d = engine().approve(intent(0.01))
    assert d.approved and d.order is not None
    assert d.order.approved_quantity == 0.01


def test_risk_rejects_oversized_position():
    d = engine(max_position_quote=100.0).approve(intent(0.01))
    assert not d.approved and d.reason == "MAX_POSITION_EXCEEDED"


def test_risk_rejects_over_exposure():
    e = engine(max_exposure_quote=400.0, max_position_quote=1e9)
    e.register_fill(300.0)
    d = e.approve(intent(0.01))
    assert not d.approved and d.reason == "MAX_EXPOSURE_EXCEEDED"


def test_leverage_limit():
    e = RiskEngine(RiskConfig(cooldown_seconds=0, max_leverage=0.5,
                              max_position_quote=1e9, max_exposure_quote=1e9),
                   RiskState(equity=1000.0))
    assert e.approve(intent(1.0, 1000.0)).reason == "LEVERAGE_EXCEEDED"


def test_daily_loss_limit_trips_kill_switch():
    e = engine(daily_loss_limit=100.0)
    e.register_fill(0.0, realized_pnl=-150.0)
    d = e.approve(intent())
    assert not d.approved and d.reason == "DAILY_LOSS_LIMIT"
    assert e.state.halted


def test_drawdown_limit():
    e = engine(max_drawdown_limit=0.1)
    e.update_equity(10_000.0)
    e.update_equity(8_000.0)
    assert e.approve(intent()).reason == "MAX_DROWDOWN".replace("DROW", "DRAW")


def test_kill_switch_blocks_everything():
    e = engine()
    e.trip_kill_switch("manual")
    d = e.approve(intent())
    assert not d.approved and "HALTED" in d.reason


def test_cooldown():
    e = RiskEngine(RiskConfig(cooldown_seconds=60), RiskState(equity=10_000.0))
    assert e.approve(intent(), now=1000.0).approved
    assert e.approve(intent(), now=1010.0).reason == "COOLDOWN"


def test_position_mismatch_stops_and_alerts():
    e = engine()
    assert e.report_position_mismatch(1.0, 1.0) == "OK"
    assert e.report_position_mismatch(1.0, 0.5) == "STOP_AND_ALERT"
    assert e.state.halted and any("POSITION_MISMATCH" in a for a in e.alerts)


def test_execution_requires_risk_approval_object():
    venue = PaperVenue()
    d = engine().approve(intent(0.01))
    fill = venue.submit(d.order)
    assert fill.status == "FILLED" and venue.position("BTCUSDT") == pytest.approx(0.01)
    with pytest.raises(AttributeError):
        venue.submit(intent(0.01))  # a raw intent is not accepted


def test_min_size_rejection():
    from trading_school_ai.execution.base import InstrumentMeta
    venue = PaperVenue(meta={"BTCUSDT": InstrumentMeta("BTCUSDT", 0.1, 0.0001, 0.001, "test")})
    d = engine().approve(intent(0.0005))
    assert venue.submit(d.order).status == "REJECTED_MIN_SIZE"


def test_precision_rounding_never_rounds_up():
    assert round_to_step(0.123456789, 0.0001) == pytest.approx(0.1234)
    assert round_to_step(1.999999, 1.0) == pytest.approx(1.0)


def test_symbol_mapping():
    assert to_okx_symbol("BTCUSDT") == "BTC-USDT"
    assert to_okx_symbol("ETH-USDT") == "ETH-USDT"
    with pytest.raises(ValueError):
        to_okx_symbol("WEIRD")


def test_live_three_locks(monkeypatch):
    monkeypatch.delenv("LIVE_CONFIRM", raising=False)
    with pytest.raises(LiveLockError) as e1:
        check_live_locks(False, False)
    assert "LOCK_1" in str(e1.value)
    with pytest.raises(LiveLockError) as e2:
        check_live_locks(True, True)
    assert "LOCK_2" in str(e2.value)
    monkeypatch.setenv("LIVE_CONFIRM", "1")
    with pytest.raises(LiveLockError) as e3:
        check_live_locks(True, False)
    assert "LOCK_3" in str(e3.value)
    check_live_locks(True, True)  # all three open


def test_okx_venue_cannot_be_built_without_locks(monkeypatch):
    monkeypatch.delenv("LIVE_CONFIRM", raising=False)
    with pytest.raises(LiveLockError):
        OKXSpotVenue(live_enabled_in_config=True, cli_live_flag=True)


def test_okx_metadata_comes_from_exchange(monkeypatch):
    monkeypatch.setenv("LIVE_CONFIRM", "1")

    class FakeClient:
        def get(self, path, params):
            assert params["instType"] == "SPOT"
            return {"data": [{"tickSz": "0.1", "lotSz": "0.00000001", "minSz": "0.00001"}]}

        def post(self, path, body):  # pragma: no cover - not used here
            raise AssertionError("no order should be sent in this test")

    venue = OKXSpotVenue(True, True, client=FakeClient())
    meta = venue.instrument("BTCUSDT")
    assert meta.inst_id == "BTC-USDT" and meta.source == "exchange"
    assert meta.tick_size == 0.1 and meta.min_size == 0.00001


def test_okx_refuses_to_guess_metadata(monkeypatch):
    monkeypatch.setenv("LIVE_CONFIRM", "1")

    class Empty:
        def get(self, path, params):
            return {"data": []}

    from trading_school_ai.execution.base import ExecutionError
    with pytest.raises(ExecutionError):
        OKXSpotVenue(True, True, client=Empty()).instrument("BTCUSDT")
