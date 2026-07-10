# Pipeline

The default stage order is:

```text
prepare -> transcribe -> correct -> align -> split -> translate -> normalize -> export
```

Only `prepare`, `transcribe`, and `export` are included in the minimal `run` path unless optional flags are enabled.

## Artifacts

- `prepare`: writes `audio.wav`, `segments.json`, and `segments/*.wav`.
- `transcribe`: reads `segments.json`, writes `transcript_segments.json` and `transcript.txt`.
- `correct`: reads `transcript_segments.json`, writes `corrected_segments.json`, and updates `transcript_segments.json`.
- `align`: reads `transcript_segments.json`, writes `aligned_segments.json`.
- `split`: reads `aligned_segments.json`, writes `split_segments.json` and `subtitles.split.srt`.
- `translate`: reads `split_segments.json`, writes `translated_segments.json` and `subtitles.translated.srt`.
- `normalize`: reads the best configured subtitle source, writes `normalized_segments.json` and `subtitles.normalized.srt`.
- `export`: reads the selected subtitle source, writes `subtitles.srt` and/or `subtitles.vtt`.

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
