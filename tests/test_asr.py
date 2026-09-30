"""ASR provider parsing and fallback, with the HTTP layer mocked (no network)."""
import httpx
import pytest

from app import asr
from app.config import Settings

WAV = b"RIFF" + b"\x00" * 40 + b"\x01\x00" * 800


class Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def settings(**kw):
    base = dict(env="test", asr_provider="google", asr_fallback="none", google_api_key="k", azure_speech_key="k", azure_speech_region="uaenorth")
    return Settings(**{**base, **kw})


def test_google_parses_transcript_confidence_and_nbest(monkeypatch):
    seen = {}

    def fake_post(url, params=None, json=None, **kw):
        seen["body"] = json
        return Resp(200, {"results": [{"alternatives": [{"transcript": "مهرج", "confidence": 0.91}, {"transcript": "مهرجه"}]}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    r = asr.transcribe(WAV, settings())
    assert r.ok and r.provider == "google" and r.transcript == "مهرج" and r.confidence == 0.91
    assert len(r.n_best) == 2
    cfg = seen["body"]["config"]
    assert cfg["languageCode"] == "ar-SA" and "speechContexts" not in cfg  # D7: no phrase hints


def test_google_empty_result_is_no_speech(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp(200, {}))
    r = asr.transcribe(WAV, settings())
    assert r.ok and r.transcript == "" and r.confidence is None


def test_azure_parses_lexical_and_confidence(monkeypatch):
    payload = {"RecognitionStatus": "Success", "NBest": [{"Confidence": 0.64, "Lexical": "ستاير", "Display": "ستاير."}]}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp(200, payload))
    r = asr.transcribe(WAV, settings(asr_provider="azure"))
    assert r.ok and r.transcript == "ستاير" and r.confidence == 0.64


def test_azure_no_match(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp(200, {"RecognitionStatus": "NoMatch"}))
    r = asr.transcribe(WAV, settings(asr_provider="azure"))
    assert r.ok and r.transcript == ""


def test_fallback_after_primary_error(monkeypatch):
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        if "googleapis" in url:
            raise httpx.ConnectTimeout("timeout")
        return Resp(200, {"RecognitionStatus": "Success", "NBest": [{"Confidence": 0.8, "Lexical": "بيت"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    r = asr.transcribe(WAV, settings(asr_fallback="azure"))
    assert r.ok and r.provider == "azure" and len(calls) == 2


def test_all_providers_fail_returns_not_ok(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp(500, {"error": "x"}))
    r = asr.transcribe(WAV, settings())
    assert not r.ok and "google" in r.error


def test_manual_provider_refused_in_prod():
    r = asr.transcribe(WAV, settings(env="prod", asr_provider="manual"), "مهرج", 0.9)
    assert not r.ok and "disabled in production" in r.error


@pytest.mark.parametrize("missing", ["google_api_key"])
def test_missing_credentials_is_an_error_not_a_crash(missing):
    r = asr.transcribe(WAV, settings(**{missing: ""}))
    assert not r.ok and "not configured" in r.error
