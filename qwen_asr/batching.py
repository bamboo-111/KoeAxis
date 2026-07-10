from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from qwen_asr.models import AudioSegment


_BUCKET_UPPER_BOUNDS = (15.0, 30.0, 60.0, 120.0, float("inf"))


@dataclass(frozen=True, slots=True)
class PlannedBatch:
    segments: list[AudioSegment]
    mode: str
    bucket_label: str
    total_duration: float
    min_duration: float
    max_duration: float
    duration_spread_ratio: float
    singleton_reason: str | None = None


class BatchPlanner:
    def __init__(
        self,
        segments: list[AudioSegment],
        *,
        mode: str,
        max_batch_items: int,
        target_audio_seconds: float | None = None,
        single_long_segment_threshold: float = 90.0,
    ) -> None:
        if mode not in {"fixed", "adaptive"}:
            raise ValueError(f"Unsupported batch mode: {mode}")
        if max_batch_items < 1:
            raise ValueError("max_batch_items must be >= 1")

        self.mode = mode
        self.max_batch_items = max_batch_items
        self.current_max_batch_items = max_batch_items
        self.remaining_segments = list(segments)
        self.single_long_segment_threshold = max(1.0, float(single_long_segment_threshold))
        self.current_target_audio_seconds = (
            float(target_audio_seconds)
            if target_audio_seconds is not None
            else self._derive_auto_target_audio_seconds(segments, max_batch_items)
        )

    def next_batch(self) -> PlannedBatch | None:
        if not self.remaining_segments:
            return None
        if self.mode == "fixed":
            segments = self.remaining_segments[: self.current_max_batch_items]
            return self._build_batch(segments, bucket_label="fixed")

        grouped = self._group_remaining_by_bucket()
        for _, bucket_segments in grouped:
            if not bucket_segments:
                continue
            if bucket_segments[0].duration >= self.single_long_segment_threshold:
                label = _bucket_label(bucket_segments[0].duration)
                return self._build_batch(
                    [bucket_segments[0]],
                    bucket_label=label,
                    singleton_reason="long_segment_threshold",
                )
            segments = self._take_adaptive_segments(bucket_segments)
            if segments:
                label = _bucket_label(segments[0].duration)
                return self._build_batch(segments, bucket_label=label)
        return None

    def mark_success(self, batch: PlannedBatch) -> None:
        completed_ids = {segment.segment_id for segment in batch.segments}
        self.remaining_segments = [
            segment
            for segment in self.remaining_segments
            if segment.segment_id not in completed_ids
        ]

    def report_oom(self, batch: PlannedBatch) -> None:
        longest = max(segment.duration for segment in batch.segments)
        self.current_max_batch_items = max(1, self.current_max_batch_items - 1)
        if self.mode == "adaptive":
            reduced_target = self.current_target_audio_seconds * 0.8
            self.current_target_audio_seconds = max(longest, round(reduced_target, 2))

    def describe_limits(self) -> str:
        return (
            f"mode={self.mode} "
            f"max_batch_items={self.current_max_batch_items} "
            f"target_audio_seconds={self.current_target_audio_seconds:.2f} "
            f"single_long_segment_threshold={self.single_long_segment_threshold:.2f}"
        )

    def _take_adaptive_segments(self, bucket_segments: list[AudioSegment]) -> list[AudioSegment]:
        selected: list[AudioSegment] = []
        total_duration = 0.0
        for segment in bucket_segments:
            if len(selected) >= self.current_max_batch_items:
                break
            next_total = total_duration + segment.duration
            if selected and next_total > self.current_target_audio_seconds:
                break
            selected.append(segment)
            total_duration = next_total
        if not selected:
            selected = [bucket_segments[0]]
        return selected

    def _build_batch(
        self,
        segments: list[AudioSegment],
        *,
        bucket_label: str,
        singleton_reason: str | None = None,
    ) -> PlannedBatch:
        total_duration = sum(segment.duration for segment in segments)
        min_duration = min(segment.duration for segment in segments)
        max_duration = max(segment.duration for segment in segments)
        return PlannedBatch(
            segments=segments,
            mode=self.mode,
            bucket_label=bucket_label,
            total_duration=round(total_duration, 2),
            min_duration=round(min_duration, 2),
            max_duration=round(max_duration, 2),
            duration_spread_ratio=round(max_duration / max(min_duration, 0.001), 3),
            singleton_reason=singleton_reason,
        )

    def _group_remaining_by_bucket(self) -> list[tuple[float, list[AudioSegment]]]:
        groups: list[tuple[float, list[AudioSegment]]] = []
        for upper_bound in _BUCKET_UPPER_BOUNDS:
            bucket = [
                segment
                for segment in self.remaining_segments
                if _bucket_upper_bound(segment.duration) == upper_bound
            ]
            groups.append((upper_bound, bucket))
        return groups

    @staticmethod
    def _derive_auto_target_audio_seconds(
        segments: list[AudioSegment],
        max_batch_items: int,
    ) -> float:
        if not segments:
            return float(max_batch_items)
        sample = median(segment.duration for segment in segments)
        per_item_budget = min(60.0, max(10.0, float(sample)))
        return round(per_item_budget * max_batch_items, 2)


def _bucket_upper_bound(duration: float) -> float:
    for upper_bound in _BUCKET_UPPER_BOUNDS:
        if duration <= upper_bound:
            return upper_bound
    return float("inf")


def _bucket_label(duration: float) -> str:
    upper_bound = _bucket_upper_bound(duration)
    if upper_bound == 15.0:
        return "0-15s"
    if upper_bound == 30.0:
        return "15-30s"
    if upper_bound == 60.0:
        return "30-60s"
    if upper_bound == 120.0:
        return "60-120s"
    return "120s+"
