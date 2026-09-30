# Naming Gaming App API (GANVA v2)

FastAPI backend for the Naming Gaming App: GANVA two-stage response classification,
server-side therapy session engine with the five-level cueing hierarchy, adaptive
response window (Algorithm S3-4), clinician flag review, blinded validation rating,
and pseudonymised exports.

> The code state before this rewrite is tagged `trial-build-as-used`. That is the
> version to cite for anything already run; this build is GANVA v2.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # edit as needed
.venv/bin/alembic upgrade head  # create / migrate the database
.venv/bin/python -m app.cli create-user --username admin --role admin
.venv/bin/python -m app.cli import-stimuli --csv seed/stimuli_s2.csv --pictures "../Gulfian Arabic/Pictures"
.venv/bin/python run.py         # http://127.0.0.1:8010, docs at /docs
```

ffmpeg is used for audio conversion; if it is not on PATH the bundled `imageio-ffmpeg` binary is used.

### Content

Sessions refuse to start until every trained item (List A, 30 items) has its picture and all
five cue levels (`python -m app.cli content-check` lists what is missing). Cue files can be
uploaded one by one in the dashboard or in bulk:

```bash
.venv/bin/python -m app.cli import-cues --dir /path/to/cues   # A-N01_L1.txt, A-N01_L1.m4a, A-N01_L2.m4a, A-N01_L3.jpg, A-N01_L5.m4a ...
```

Item pictures can be included as `A-N01_picture.jpg`, and an `associates.csv` in the same folder is imported too.

For development only, `python -m app.cli demo-content` fills missing cues with tones, a placeholder picture and
"[DEMO" texts. The content check recognises these placeholders (and an L3 visual cue that is just a copy of the
item picture) and treats them as missing, so sessions stay blocked until real content replaces them.
`REQUIRE_COMPLETE_CONTENT=false` lets development sessions run anyway; never use it for clinical work.

To move content to another server (e.g. the cloud deployment), export the real content and import it there:

```bash
.venv/bin/python -m app.cli export-content --dir ./content-export     # skips demo placeholders; writes manifest.csv with checksums
.venv/bin/python -m app.cli import-cues --dir ./content-export        # on the target server, after import-stimuli
```

`seed/stimuli_s2.csv` is generated from Supplementary Table S2-2 by `scripts/extract_s2_stimuli.py`.

## Study parameters

All protocol values (GANVA thresholds, adaptive controller, session rules, points) live in
`app/config.py` and are fingerprinted by a config hash stamped on every session and attempt
(`GET /api/config`). Changing one is a reviewed code change, never a runtime setting.

| Area | Value |
|---|---|
| Normalisation (S3-1) | NFD, drop Mn; أ إ آ ٱ → ا; ة → ه; ى ئ → ي; remove tatweel, zero-width marks, punctuation; collapse spaces |
| Similarity (S3-2) | normalised Levenshtein; best matching word/n-gram of the transcript (D5) |
| Classification (S3-3) | ≥ 0.70 ACCEPT; < 0.50 REJECT; 0.50–0.70 → semantic associates (≥ 0.70 to any → ACCEPT); ASR confidence < 0.70, missing, or ASR error → FLAG |
| Adaptive window (S3-4) | start mild 30 / moderate 45 / severe & profound 60 s; last 3 trials; updated after every trial; −5 s if accuracy > 70 % and mean response < 0.6 × window; +10 s if time-outs > 30 %; +5 s if accuracy < 50 %; clamp 20–60 s |
| Session | all 30 trained items in seeded random order per pass; passes repeat until 60 active minutes, then the current item finishes |
| Cues | 0 cue-free → 1 semantic → 2 phonemic → 3 visual → 4 written word → 5 full auditory model; after an L5 failure one repetition, then the item is scored incorrect |
| FLAG in session | the supervising SLP taps Correct/Incorrect; after 20 s the default (REJECT) applies and the flag stays in the review queue |

## ASR

`ASR_PROVIDER=google` (Cloud Speech-to-Text v1 REST, API key) or `azure` (short-audio REST, detailed
format), optional `ASR_FALLBACK`. The language is `ASR_LOCALE`, default `ar-SA` (Saudi Arabic, for the Najdi group). No phrase hints are sent (D7). `manual` lets the client send the
transcript and confidence for development and automated tests, and is refused when `ENV=prod`.
Confirm with a pilot recording that the chosen locale/model returns a confidence for single words.

## Security

- Passwords: argon2. Auth: JWT (access + refresh), roles `admin`, `clinician`, `rater`, `patient`.
- Participant audio is stored only with `consent_audio_retention`, encrypted with AES-256-GCM (`AUDIO_KEY_B64`).
- `JWT_SECRET` and `AUDIO_KEY_B64` are required when `ENV=prod`.
- For encryption of the database itself use PostgreSQL on an encrypted volume (`DATABASE_URL`) in production.
- Exports carry study codes only (no usernames, notes or audio references). Access, decisions and exports are audit-logged.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Covers normalisation, Levenshtein against a reference implementation, every classifier branch,
every adaptive rule, full session flows (accept, cue ladder, repetition, FLAG with clinician and
default resolution, time-outs, duration end, idempotent uploads), encryption at rest, role access,
exports and the validation metrics.

## Main endpoints

| | |
|---|---|
| `POST /api/auth/login`, `/api/auth/refresh`, `GET /api/auth/me` | sign in |
| `GET/POST /api/participants`, `GET/PATCH /api/participants/{id}`, `/sessions`, `/adaptive`, `/gamification` | participants |
| `GET /api/stimuli`, `/content-check`, `PUT /api/stimuli/{id}/picture`, `PUT /api/stimuli/{id}/cues/{level}`, associates, `POST /api/stimuli/associates/import` | content |
| `POST /api/sessions` (start or resume), `GET /{id}/state`, `/pause`, `/resume`, `/end` | sessions |
| `POST /api/sessions/{id}/trials/{trial}/response` (multipart audio), `/timeout` | responses |
| `POST /api/sessions/{id}/attempts/{a}/decide`, `/default`, `/feedback` | flags in session, latency |
| `GET /api/sessions/{id}/analytics`, `/attempts` | analytics |
| `GET /api/flags`, `POST /api/flags/{a}/review`, `GET /api/attempts/{a}/audio` | flag review |
| `/api/validation/samples`, `/next?rater_no=`, `/ratings`, `/metrics` | blinded validation |
| `GET /api/export/{name}.csv`, `GET /api/config` | export, configuration |
