from app.adaptive import TrialStat, initial_limit_ms, next_limit
from app.config import AdaptiveConfig

CFG = AdaptiveConfig()


def T(correct=True, rt=5_000, timed_out=False):
    return TrialStat(correct, rt, timed_out)


def test_initial_limits():
    assert initial_limit_ms("mild", CFG) == 30_000
    assert initial_limit_ms("Moderate", CFG) == 45_000
    assert initial_limit_ms("severe", CFG) == 60_000
    assert initial_limit_ms("profound", CFG) == 60_000
    assert initial_limit_ms(None, CFG) == 45_000


def test_insufficient_history():
    r = next_limit(45_000, [T(), T()], CFG)
    assert (r.rule, r.new_ms) == ("insufficient_history", 45_000)


def test_decrease_when_accurate_and_fast():
    r = next_limit(45_000, [T(rt=10_000)] * 3, CFG)  # acc 1.0, 10 s < 27 s
    assert (r.rule, r.new_ms) == ("decrease", 40_000)


def test_accurate_but_slow_is_no_change():
    r = next_limit(45_000, [T(rt=30_000)] * 3, CFG)  # 30 s >= 0.6*45 s
    assert r.rule == "no_change" and r.new_ms == 45_000


def test_hard_increase_on_timeouts():
    hist = [T(False, 45_000, True), T(True, 8_000), T(True, 8_000)]  # timeout 1/3 > 0.30
    r = next_limit(45_000, hist, CFG)
    assert (r.rule, r.new_ms) == ("hard_increase", 55_000)


def test_soft_increase_on_low_accuracy():
    hist = [T(False, 20_000), T(False, 20_000), T(True, 20_000)]  # acc 0.33 < 0.5, no timeouts
    r = next_limit(45_000, hist, CFG)
    assert (r.rule, r.new_ms) == ("soft_increase", 50_000)


def test_timeouts_checked_before_low_accuracy():
    hist = [T(False, 45_000, True)] * 3
    assert next_limit(45_000, hist, CFG).rule == "hard_increase"


def test_boundaries_are_strict():
    # acc exactly 0.70 is not > 0.70; can't hit exactly with n=3, so use n=10 window
    cfg = CFG.model_copy(update={"window_n": 10, "min_history": 10})
    hist = [T(True, 1_000)] * 7 + [T(False, 1_000)] * 3
    assert next_limit(45_000, hist, cfg).rule == "no_change"
    # acc exactly 0.50 is not < 0.50
    hist = [T(True, 20_000)] * 5 + [T(False, 20_000)] * 5
    assert next_limit(45_000, hist, cfg).rule == "no_change"


def test_clamped_to_20_60():
    assert next_limit(20_000, [T(rt=1_000)] * 3, CFG).new_ms == 20_000
    assert next_limit(60_000, [T(False, 60_000, True)] * 3, CFG).new_ms == 60_000
    assert next_limit(55_000, [T(False, 60_000, True)] * 3, CFG).new_ms == 60_000


def test_uses_newest_first_window_only():
    newest = [T(rt=1_000)] * 3
    older = [T(False, 60_000, True)] * 5
    r = next_limit(45_000, newest + older, CFG)
    assert r.rule == "decrease" and r.n == 3
