"""
Extract the 60-item stimulus set (Table S2-2a-d) from the Supplementary Material
docx into seed/stimuli_s2.csv.

    python scripts/extract_s2_stimuli.py "../04_Supplementary_Material_S1-S6.docx"
"""
from __future__ import annotations

import csv
import re
import sys
import zipfile
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
OUT = Path(__file__).resolve().parent.parent / "seed" / "stimuli_s2.csv"
HEADER = ["No", "English", "Arabic (Gulf)", "Gen/Trans", "Syll", "Phon", "Freq", "RT (ms)", "NA", "VC", "IA", "Imag", "Fam", "AoA", "Phonemic cue", "Semantic cue"]
# Table order in S2-2: a = List A nouns, b = List A verbs, c = List B nouns, d = List B verbs
TABLE_META = [("A", "noun"), ("A", "verb"), ("B", "noun"), ("B", "verb")]


def tables(docx: Path) -> list[list[list[str]]]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(zipfile.ZipFile(docx).read("word/document.xml"))
    out = []
    for tbl in root.iter(f"{W}tbl"):
        rows = []
        for tr in tbl.iter(f"{W}tr"):
            cells = []
            for tc in tr.iter(f"{W}tc"):
                cells.append(" ".join("".join(t.text or "" for t in p.iter(f"{W}t")) for p in tc.iter(f"{W}p")).strip())
            rows.append(cells)
        out.append(rows)
    return out


def main(docx: str) -> None:
    stim_tables = [t for t in tables(Path(docx)) if t and len(t[0]) == 16 and t[0][1] == "English" and "Phonemic cue" in t[0]]
    if len(stim_tables) != 4:
        sys.exit(f"expected 4 stimulus tables, found {len(stim_tables)}")
    rows = []
    for (word_list, word_class), tbl in zip(TABLE_META, stim_tables):
        for r in tbl[1:]:
            no = r[0]
            num = int(re.sub(r"\D", "", no))
            rows.append(
                {
                    "code": f"{word_list}-{word_class[0].upper()}{num:02d}",
                    "list": word_list,
                    "word_class": word_class,
                    "number": num,
                    "frequency_source": "ArabiCorpus" if "‡" in no else "ARALEX",
                    "english": r[1],
                    "target": r[2],
                    "grammatical": r[3],
                    "syllables": r[4],
                    "phonemes": r[5],
                    "frequency_pm": r[6],
                    "rt_ms": r[7],
                    "name_agreement": r[8],
                    "visual_complexity": r[9],
                    "image_agreement": r[10],
                    "imageability": r[11],
                    "familiarity": r[12],
                    "aoa": r[13],
                    "phonemic_cue_label": r[14],
                    "semantic_cue_en": r[15],
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../04_Supplementary_Material_S1-S6.docx")
