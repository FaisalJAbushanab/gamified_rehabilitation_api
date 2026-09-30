import pytest

from app.validation import cohen_kappa, confusion, kappa_with_ci


def test_kappa_hand_computed():
    # po = .75, pe = .5*.25 + .5*.75 = .5 -> kappa = .5
    a = [True, True, False, False]
    b = [True, False, False, False]
    assert cohen_kappa(a, b) == pytest.approx(0.5)


def test_kappa_ci_contains_estimate():
    a = [True] * 40 + [False] * 40
    b = [True] * 36 + [False] * 4 + [False] * 38 + [True] * 2
    r = kappa_with_ci(a, b)
    assert r["ci95"][0] <= r["kappa"] <= r["ci95"][1]
    assert r["agreement"] == pytest.approx(74 / 80, abs=1e-4)


def test_confusion_matches_s3_definitions():
    # (ganva_positive, human_positive)
    pairs = [(True, True)] * 8 + [(True, False)] * 2 + [(False, True)] * 1 + [(False, False)] * 9
    c = confusion(pairs)
    assert (c["tp"], c["fp"], c["fn"], c["tn"], c["n"]) == (8, 2, 1, 9, 20)
    assert c["precision"] == 0.8 and c["recall"] == pytest.approx(0.889, abs=1e-3)
    assert c["specificity"] == pytest.approx(9 / 11, abs=1e-3) and c["accuracy"] == 0.85
