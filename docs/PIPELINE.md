# Pipeline

The registered stage order is:

```text
prepare -> transcribe -> correct -> align -> split -> translate -> mimo-proofread -> proofread-realign -> quality-gate -> normalize -> export
```

The minimal `run` path is `prepare -> transcribe -> quality-gate -> export`. Optional flags insert `correct`, `align`, `split`, `translate`, MiMo proofreading plus proofread realignment, and `normalize` in the registered order. The quality gate always runs before formal export and a failing gate stops the pipeline.

Production defaults use silence-first segmentation with a 15 second maximum segment window, project-local `.model-cache`, `local_files_only` enabled unless explicitly disabled, adaptive ASR batching, and stage checkpoints for resumable `transcribe` and `align`. The CLI leaves `batch_size` unset initially; adaptive resolution selects 3, 4, or 5 from the current segment-duration distribution, while fixed mode falls back to 5 when no explicit size is supplied.

When a 15 second window contains no VAD silence of at least `--min-silence-ms` (continuous talk over BGM, typical for radio), prepare no longer cuts mid-word at the hard limit: it searches the last 40% of the window for the quietest 200 ms point of the source audio and cuts there if it is at least 10 dB below the window median, padding that boundary by only 80 ms so neighbouring segments do not re-transcribe the same words. Windows without such a dip keep the original hard cut. On the 33-minute episode used for validation this moved 71 of 99 hard cuts into pauses and removed the name corruption caused by a cut inside `リナたそ`.

Glossary columns whose Chinese name cells carry a kana reading, for example `声优/乐手名（中文）` with `理名(りな)`, contribute `りな`/`リナ => 理名` prompt entries. MiMo proofread suggestions that would put Japanese kana back into a kana-free translation are rejected (`translation-introduces-japanese-kana`), and translator output lines that echo the instruction (`翻译为：…`) are stripped before validation.

If a translation batch is still invalid after the maximum attempts but is structurally complete (all keys present, every line translated) and only some lines keep Japanese kana, the batch is kept and those lines are flagged as translation suspects instead of dropping the whole batch. Missing keys or empty translations still fail the batch.

Normalize keeps every cue visible for at least `min_duration_ms` (500 ms): non-positive cues are merged first, a too-short cue may start earlier inside the preceding free gap, and as a last resort borrows display time from the previous cue. Subtitle layout never treats a dot followed by an ASCII letter or digit as a sentence end, and display punctuation cleanup keeps domain dots, so spoken addresses such as `土下座 at interfm.jp` are neither split across cues nor stripped to `interfm jp`.

MiMo prompt cost: prompt JSON is compact UTF-8 (`ensure_ascii=False`; escaped CJK used several times the tokens), the segment is reduced to id, span, language and text, and the glossary sent per call holds only entries that occur in the source or translation plus name groups and katakana show terms; other usage notes are sent only on a textual hit, with short notes and no group label. Lines whose suspect types are only `short_response`, `fragment` or `semantic` stay in the suspect report but are not sent to nearby-audio review (on four radio episodes 464 such calls produced 4 punctuation-level edits); the final MiMo checkpoint counts pending audio reviews with the same rule. Batch adjacent suspects with `--mimo-nearby-batch-size` (3 is used for radio). On the next episode this cut stage-2 prompt tokens from 4.48M to 0.62M.

Qwen ForcedAligner is the only production main aligner. Split has one production implementation, `rule`; the rejected LLM split variants and MFA full-alignment backend are absent from CLI, Web, and PipelineRunner. MFA local remains an explicit, default-off local fallback for applicable Japanese proofread or failed-segment recovery candidates; it is never a full-dataset replacement. Translation LLM, MiMo, the generic LLM client, timing legality checks, and the mandatory pre-export quality gate remain supported.

After Align, `recover-align` and the Web recovery layer route every failed dialogue segment through the same executor. Music regions are represented separately as `SKIPPED_MUSIC_REGION`; they remain inspectable but are excluded from dialogue recovery counts and the live alignment quality gate. Qwen retry uses the original transcript unless verified text is explicitly opted in. `route_language` changes actual backend selection, and MFA local is Japanese-only. Exact results pass the same token timing/content validator regardless of backend.

VAD is localization evidence, not transcript authority. A recovery action that accepts `completed_coarse` requires verified text plus one unique or manually selected VAD region inside the original segment, bounded duration for short responses, and safe neighbor overlap. It first backs up aligned manifest/checkpoint/event evidence, atomically updates shared alignment state, records the original failure and operator, and leaves the prior quality FAIL visible until quality is rerun. Multiple ambiguous regions or no speech preserve `failed`.

Web cue editing writes only `drafts/web-review.json`. Each edit validates cue identity, non-empty source text, positive timing, non-overlap, and expected revision; it backs up an existing draft and appends `reports/web_review_history.jsonl`. A dirty review draft invalidates quality/normalize/export presentation without silently replacing `normalized_segments.json` or exported subtitles. Undo restores the previous draft cue while preserving formal manifests and audit history.

## Artifacts

- `prepare`: writes `audio/source.wav`, `segments.json`, and optionally `audio/segments/*.wav` when eager segment export is enabled.
- `transcribe`: reads `segments.json`, writes `transcript_segments.json`, `transcript.txt`, `transcript_events.jsonl`, `transcript_checkpoint.json`, and optional `transcribe.profile.json`. Optional `--asr-context-terms "a,b"` / `--asr-context-file terms.txt` (off by default; example `samples/asr-context-togeradi.txt`) passes Qwen3-ASR a bare `用語: a, b, …` context. Keep it to fixed-reading band/station/show/corner names: on radio episode 46 it fixed トゲナシトゲアリ (2→21), InterFM (0→5) and ガールズバンドクライ (0→3) at unchanged speed, while member names and e-mail addresses did not improve and sentence-style prompts were copied into 10 segments. Any segment whose output contains the `用語:` header or a run of three or more prompt terms separated by commas/読点 is re-transcribed without context and listed under `context_leak_retranscribed` in the transcribe profile. The prompt also perturbs decoding away from the terms (episodes 43/44: ~55% of segments changed, mostly unrelated words such as アイス→愛して, and a blind MiMo listening judge only agreed with itself on 63% of those), so the default `--asr-context-mode merge` transcribes every batch with and without context and keeps the no-context text except for edits that land on a prompt term (character diff); this kept the term fixes (トゲナシトゲアリ 2→13 / 0→15, InterFM 0→3 / 0→5) while segments differing from the plain transcript fell from 127/159 to 18/19, at the cost of a second ASR pass (~2 min per 40-minute episode). Merged segments are listed under `context_term_edits`; `--asr-context-mode full` uses the context transcript as is.
- `correct`: reads `transcript_segments.json`, writes `corrected_segments.json`, and updates `transcript_segments.json`.
- `align`: reads `transcript_segments.json`, writes `aligned_segments.json`, `aligned_events.jsonl`, and `aligned_checkpoint.json`. Coverage is judged against the shorter of the VAD segment and the time the text needs at 6 chars/s, so a short line after a jingle or a lone interjection in a long segment is no longer rejected. When only a few tokens collapse to ~0s (local density error), those tokens are re-timed between their trustworthy neighbours (widening by up to 3 neighbours if the gap is too tight; a wide gap only gets the time 8 chars/s needs, centred on the collapse point) instead of discarding the whole segment; repair is refused when more than 35% of the characters collapsed or the result still fails validation. Repaired segments stay `completed_exact` with `recovery.method = "align-local-repair"` and the repaired spans; translate marks overlapping subtitles `suspect_types: ["alignment_repaired"]` + `needs_audio_review`, because collapsed text is often not actually spoken. On radio episodes 45–50 this turned 32 of 36 rejected segments into exact alignments.
- `recover-align`: reads one failed entry from aligned state, invokes Qwen or applicable Japanese MFA local, and updates only that segment after backup and shared exact validation; failed attempts preserve the original state. MFA start-up dominates its cost (each `mfa align --clean` spends ~110 s loading the Japanese dictionary/tokenizer, aligning takes seconds), so batch callers first run `prefetch_alignment_recovery_mfa(work_paths)`: every currently failed Japanese segment is aligned in ONE MFA call (one speaker directory per clip, so speaker adaptation and word timings match individual runs) and cached in `experiments/mfa-align-experiment/batch-cache.json`, keyed by audio file (size+mtime), padded clip range and lab text. The per-segment `mfa-local` attempt then reads the cache (~2.5 s) and still applies every guard, backup and writeback; a cache miss (other text/range) runs the usual individual `--clean` call. Never drop `--clean` to save time: MFA caches by corpus directory name and silently returns the previous clip's words. Proofread realign with `--proofread-realign-primary mfa-local` prefetches its candidates the same way. Replaying episodes 43/44 recovery: identical outcomes for all 7 segments (one word boundary moved 10 ms), 4.4 min -> 2.5 min and 8.9 min -> 2.9 min.
- `mfa`: optional local proofread fallback plus standalone experiment tooling; MFA environments, corpora, pretrained models, and micromamba files are local state under `tools/` and are not tracked as source.
- `split`: rule-only production stage; reads `aligned_segments.json`, writes `split_segments.json` and `subtitles.split.srt`. With `--split-boundaries bunsetsu` (default; needs `fugashi` + `unidic-lite`, otherwise falls back to unrestricted cuts) Japanese long-line cuts and time-gap anomaly cuts are only placed at bunsetsu boundaries (auxiliaries, particles, て+補助動詞, compound nouns and サ変 stay attached), still choosing the largest timing gap among allowed positions; a pause inside a bunsetsu only forces a cut above 1.5 s. On episodes 45–50 heuristic bad cuts dropped from 508 to 144 and ≤4-char fragments from 587 to 498. When an LLM is configured, an optional semantic merge then joins adjacent cues the model groups as one phrase; code guards still refuse merges over 36 chars, 10 s, a 1 s gap, a sentence end or a standalone response, but a rejected 3–4 cue chain is now split into its longest guard-passing runs instead of being dropped entirely (+5–13 merges per episode on 46/47). With a MiMo `--llm-model` this step defaults to `mimo-v2.6-flash` (`--split-llm-model` overrides): on 46/47 Flash merged ~1.5–2× as many half-sentences as Pro with no failed batches, at the cost of more lines over 24 chars. Each merge batch is retried twice (1 s, 2 s back-off); a batch that still fails (network stall, invalid JSON, coverage guard) keeps its rule-split cues with a warning instead of failing the whole stage (episode 51 once lost its split stage to a single request that stalled for ~6 minutes). Auth errors still fail fast.
- `translate`: reads `split_segments.json`, writes `translated_segments.json` and `subtitles.translated.srt`. Output is the slow part (the model emits ~60 tokens/s), so lines with nothing to flag (confidence >= 0.8) return only `{"translation", "confidence"}` and the four suspect fields are written only for flagged lines; missing fields mean "not suspect". On episode 43 this cut output tokens 64.7k -> 34.1k and wall time 184 s -> 118 s (67 s at concurrency 10) with the same coverage of the lines audio review later corrected (13/15). Returning a bare string for clean lines was tried and rejected: output fell to 13.9k tokens but translations dropped content (e.g. a line reduced to its first word). Default LLM/MiMo concurrency is 10 (was 5): at 10 the translate requests showed no 429s or retries and per-request latency stayed ~10 s.
- `proofread`: optional MiMo suspects-only flow; reads candidate/suspect inputs, writes checkpointed batch state, evidence reports, and protected edit outputs without changing the default subtitle schema.
- `mimo-proofread` model selection: the proofread subprocess now always receives `--model` (`--mimo-model`, else `--llm-model` when it is a MiMo model, else the project default `mimo-v2.6-pro`). Since the text LLM default moved to `gemini-3.8-flash` (2026-10), a non-MiMo `--llm-model` is never forwarded to the MiMo endpoint. Earlier builds passed no model and the script silently used `mimo-v2.5` for every audio review request, which was the bulk of the MiMo token spend. `--mimo-text-model` (stage1 text screening, skipped when translate already wrote suspect metadata) and `--mimo-audio-model` (nearby/segment audio review) override it per task; `--split-llm-model` does the same for the split semantic-merge requests (default Flash for MiMo, see `split`). The models used are printed at start (`mimo models: text=… audio=…`) and recorded per request in `reports/pipeline_performance.api.json`.
- Text LLM transport: `optimizer/gemini_native.py` serves `call_llm` when the base URL contains `/v1beta`/`/v1alpha`, is a `generativelanguage.googleapis.com` URL or uses the `gemini+https://` prefix. It maps system/user/assistant messages, JSON mode, `max_tokens` and MiMo's `{"thinking":{"type":"disabled"}}` (→ `thinkingLevel: low`) to `generateContent`, passes only known top-level Gemini fields from `--llm-extra-body-json` (e.g. `{"generationConfig":{"thinkingConfig":{"thinkingLevel":"high"}}}` to raise thinking), and returns an OpenAI-shaped `ChatCompletion`, so translate/split/correct and performance accounting are unchanged. Keys come from `GEMINI_API_KEY` for native URLs or the `GEMINI_BASE_URL` host.
- `quality`: optional quality gates; content, ASS, diff, final quality, and proofread-realign checks write reports while preserving source segment schema.
- `normalize` program memory: optional `--program-memory-file rules.json` (on `run`, `batch` and `normalize`; off by default; example `samples/program-memory-togeradi.json`, module `qwen_asr/program_memory.py`). Right after normalize writes its output, per-show regex rules are applied cue by cue to `normalized_segments.json` and `subtitles.normalized.srt` is rewritten. It runs there, not on the proofread manifest, because normalized cues are whole display sentences: in the proofread manifest a title is often split across cues (episode 53: `赤色の` / `共済を`), so the rules saw far less. Rules never change timing or cue count; a translation rule can require the corrected source line (`when_source`) so it only fires on the right cue. `flags` only report cues that need a human (e.g. a venue name split across cues). Results go to `reports/program_memory.json`, including per-cue `source_reverts` (Japanese before/after); the normalize content check undoes exactly those cues, so these deliberate fixes do not count as "normalize changed Japanese" while any other change still fails. When text changes, a previously exported `export-cache/subtitles.srt`/`.vtt` is deleted (it no longer matches; export rewrites it). Rules are idempotent. It covers the recurring radio misrecognitions that the ASR context prompt does not fix: the station name before the band name (インターフェム / インタFM / `interfm.jp` → InterFM), the concert title (赤色の共済/競賽/共催 → 赤色の響祭, 《赤色响祭》), the venue (ピアリーナMM → ぴあアリーナMM / PIA ARENA MM) and garbled mail addresses (→ togeradi@interfm.jp). On the raw outputs of episodes 24 and 43–50 it changed about 120 cues with no wrong edit found on review.
- `normalize`: reads the best configured subtitle source, repairs conservative bilingual display boundaries without changing wording, removes display punctuation (preserving decimal/URL/email/word spelling; the katakana middle dot ・ counts as punctuation in the content checks too), and writes `normalized_segments.json`, `subtitles.normalized.srt`, and `reports/normalize_layout.json`. Paired sentence splits use aligned token boundaries where available; proportional display timing is explicitly recorded otherwise. Standalone short responses remain separate and use available neighboring display time toward a 500ms minimum.
- `export`: reads the selected subtitle source, writes `subtitles.srt` and/or `subtitles.vtt`.
- `web recovery`: writes recovery state/action evidence below `reports/`; real retry, exact success, guarded coarse acceptance, and target-level undo all use backed-up aligned state through the shared recovery executor/service.
- `web review`: writes `drafts/web-review.json`, `reports/web_review_history.jsonl`, and `reports/review-backups/`; it does not overwrite formal manifests.

## Progress

`progress.json` uses these stable fields:

```json
{
  "stage": "translate",
  "status": "running",
  "done": 3,
  "total": 8,
  "current": "batch 1",
  "updated_at": "2026-06-06T00:00:00Z",
  "summary": "3/8 translated subtitles"
}
```

Status values are `running`, `completed`, `failed`, and `skipped`.

## Local Verification

Use the project environment when available:

```powershell
.\scripts\local_check.ps1
```

The script runs `compileall`, the full pytest suite, Ruff, and `git diff --check`. Install the pinned development dependencies with `.venv312\Scripts\python.exe -m pip install -r requirements-dev.txt` before running it. `-SkipRuff` is diagnostic-only and is not accepted for final verification.

## Rebuilding The Optional MFA Environment

MFA is local experimental state and is not required for the default Qwen alignment path. The currently verified local versions are micromamba `2.8.1` and Montreal Forced Aligner `3.4.0`, with the `japanese_mfa` acoustic model and dictionary.

From the repository root on a new Windows machine:

```powershell
New-Item -ItemType Directory -Force tools\micromamba\extract, tools\mfa-root | Out-Null
Invoke-WebRequest https://micro.mamba.pm/api/micromamba/win-64/latest `
  -OutFile tools\micromamba\micromamba.tar.bz2
tar.exe -xjf tools\micromamba\micromamba.tar.bz2 -C tools\micromamba\extract

$mamba = Resolve-Path tools\micromamba\extract\Library\bin\micromamba.exe
& $mamba create -y -p tools\mfa-env -c conda-forge "montreal-forced-aligner=3.4.0"

$env:MFA_ROOT_DIR = (Resolve-Path tools\mfa-root).Path
& $mamba run -p tools\mfa-env mfa model download acoustic japanese_mfa
& $mamba run -p tools\mfa-env mfa model download dictionary japanese_mfa
& $mamba run -p tools\mfa-env mfa version
```

The final command must print `3.4.0`. Runtime discovery checks the micromamba environment first, then `tools\mfa-env\Scripts\mfa.exe`, then an `mfa` executable on `PATH`. `MFA_ROOT_DIR` should continue to point to `tools\mfa-root` so downloaded models and corpora remain project-local and ignored by Git.

These commands require network access to micromamba/conda-forge and the MFA model registry. Existing `tools\mfa-env`, `tools\mfa-root`, and `tools\micromamba` directories must not be overwritten, moved, or deleted merely to test the instructions.
