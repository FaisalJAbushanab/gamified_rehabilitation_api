"""
ASR provider layer.

Every provider returns ``ASRResult`` with a transcript and, crucially for GANVA,
a confidence value. Phrase hints / target-word boosting are deliberately not
used (decision D7): they would bias recognition towards ACCEPT.

Providers
- ``google``: Google Cloud Speech-to-Text v1 REST (``speech:recognize``), top alternative confidence.
- ``azure``: Azure Speech short-audio REST, ``format=detailed``, NBest[0].Confidence, Lexical form.
- ``manual``: development/testing only; the transcript and confidence are sent by the
  client in the form fields ``dev_transcript`` / ``dev_confidence``. Refused when ENV=prod.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import httpx

from .config import Settings


@dataclass
class ASRResult:
    ok: bool
    provider: str
    model: str = ""
    locale: str = ""
    transcript: str | None = None
    confidence: float | None = None
    n_best: list[dict] = field(default_factory=list)
    error: str = ""
    latency_ms: float = 0.0


class ASRError(Exception):
    pass


def _google(wav: bytes, s: Settings) -> ASRResult:
    if not s.google_api_key:
        raise ASRError("GOOGLE_API_KEY is not configured")
    body = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "languageCode": s.asr_locale,
            "maxAlternatives": 5,
            "enableAutomaticPunctuation": False,
            "model": "default",
        },
        "audio": {"content": base64.b64encode(wav[44:]).decode()},
    }
    r = httpx.post(
        "https://speech.googleapis.com/v1/speech:recognize",
        params={"key": s.google_api_key},
        json=body,
        timeout=s.asr_timeout_s,
    )
    if r.status_code != 200:
        raise ASRError(f"google http {r.status_code}: {r.text[:200]}")
    results = r.json().get("results", [])
    if not results:
        return ASRResult(True, "google", "default", s.asr_locale, "", None, [])
    alts = results[0].get("alternatives", [])
    n_best = [{"transcript": a.get("transcript", ""), "confidence": a.get("confidence")} for a in alts]
    # Multiple results = multiple utterance segments; join their top transcripts.
    transcript = " ".join(res["alternatives"][0].get("transcript", "") for res in results if res.get("alternatives")).strip()
    return ASRResult(True, "google", "default", s.asr_locale, transcript, alts[0].get("confidence") if alts else None, n_best)


def _azure(wav: bytes, s: Settings) -> ASRResult:
    if not (s.azure_speech_key and s.azure_speech_region):
        raise ASRError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not configured")
    url = f"https://{s.azure_speech_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
    r = httpx.post(
        url,
        params={"language": s.asr_locale, "format": "detailed", "profanity": "raw"},
        headers={
            "Ocp-Apim-Subscription-Key": s.azure_speech_key,
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
            "Accept": "application/json",
        },
        content=wav,
        timeout=s.asr_timeout_s,
    )
    if r.status_code != 200:
        raise ASRError(f"azure http {r.status_code}: {r.text[:200]}")
    data = r.json()
    status = data.get("RecognitionStatus")
    if status in ("NoMatch", "InitialSilenceTimeout", "BabbleTimeout"):
        return ASRResult(True, "azure", "conversation", s.asr_locale, "", None, [])
    if status != "Success":
        raise ASRError(f"azure status {status}")
    nbest = data.get("NBest", [])
    n_best = [{"transcript": n.get("Lexical", ""), "confidence": n.get("Confidence"), "display": n.get("Display", "")} for n in nbest]
    top = n_best[0] if n_best else {"transcript": "", "confidence": None}
    return ASRResult(True, "azure", "conversation", s.asr_locale, top["transcript"], top["confidence"], n_best)


def _manual(dev_transcript: str | None, dev_confidence: float | None, s: Settings) -> ASRResult:
    if s.env == "prod":
        raise ASRError("manual ASR provider is disabled in production")
    return ASRResult(True, "manual", "dev", s.asr_locale, dev_transcript or "", dev_confidence, [])


def transcribe(wav: bytes, s: Settings, dev_transcript: str | None = None, dev_confidence: float | None = None) -> ASRResult:
    """Run the primary provider, then the fallback on error. Never raises."""
    order = [s.asr_provider] + ([s.asr_fallback] if s.asr_fallback not in ("none", s.asr_provider) else [])
    errors = []
    for name in order:
        t0 = time.perf_counter()
        try:
            if name == "google":
                res = _google(wav, s)
            elif name == "azure":
                res = _azure(wav, s)
            else:
                res = _manual(dev_transcript, dev_confidence, s)
            res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return res
        except (ASRError, httpx.HTTPError, ValueError, KeyError) as e:
            errors.append(f"{name}: {e}")
    return ASRResult(False, order[0], locale=s.asr_locale, error="; ".join(errors))
