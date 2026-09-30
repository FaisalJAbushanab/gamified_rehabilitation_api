"""
Algorithm S3-1: Arabic text normalisation.

Applied to the ASR transcript, the target and every semantic associate before
any similarity computation.

Note on step order: NFD decomposition splits أ إ آ ئ ؤ into a base letter plus a
combining hamza/madda (category Mn), so step 2 already reduces them to ا ا ا ي و.
The explicit mappings in steps 3-5 keep the result correct for input that
arrives already decomposed or uses the separate alef wasla (U+0671), which has
no decomposition.
"""
from __future__ import annotations

import re
import unicodedata

_ALEF_VARIANTS = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
_TA_MARBUTA = {"ة": "ه"}
_YEH_VARIANTS = {"ى": "ي", "ئ": "ي"}
_TATWEEL = "ـ"
_ZERO_WIDTH = re.compile("[​-‏⁦-⁩﻿]")
# Arabic and Latin punctuation ASR engines emit (، ؛ ؟ . , ! ? ; : " ' etc.).
_PUNCT = re.compile(r"[،؛؟۔.,!?;:\"'()\[\]{}\-…«»]")
_SPACES = re.compile(r"\s+")

_MAP = str.maketrans({**_ALEF_VARIANTS, **_TA_MARBUTA, **_YEH_VARIANTS})


def normalize_arabic_text(text: str | None) -> str:
    if not text:
        return ""
    # 1-2. NFD, then drop every combining mark (harakat, shadda, sukun, dagger alef, hamza/madda marks)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    # 3-5. alef variants -> ا, ta marbuta -> ه, alef maqsura / yeh-hamza -> ي
    text = text.translate(_MAP)
    # Additions documented in S3 (D6): tatweel, zero-width/direction marks, punctuation
    text = text.replace(_TATWEEL, "")
    text = _ZERO_WIDTH.sub("", text)
    text = _PUNCT.sub(" ", text)
    # 6. trim and collapse internal whitespace
    return _SPACES.sub(" ", text).strip()
