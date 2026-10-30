"""Fixed synthetic rows for the offline rules reference. Not the 149-segment raw."""

def segment(segment_id, text, status="completed", error=None):
    return {
        "segment_id": segment_id,
        "audio_path": "readonly/not-copied.wav",
        "global_start_time": 0.0,
        "global_end_time": 1.25,
        "text": text,
        "language": "Japanese",
        "raw_model_output": {"source": "synthetic"},
        "status": status,
        "error": error,
    }


CASES = [
    {"id": "changed", "rows": [segment("c1", "棘ラジ")]},
    {"id": "unchanged", "rows": [segment("u1", "そのままです")]},
    {"id": "failed", "rows": [segment("f1", "棘ラジ", status="failed", error="asr-failed")]},
    {"id": "empty", "rows": [segment("e1", "")]},
    {"id": "blank", "rows": [segment("b1", "   ")]},
    {"id": "mixed", "rows": [
        segment("m1", "棘ラジ"),
        segment("m2", "そのままです"),
        segment("m3", "棘ラジ", status="failed", error="asr-failed"),
    ]},
]
