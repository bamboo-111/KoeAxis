"""Opt-in, redacted performance events for a real-chain run.

The caller owns instrumentation. No model, endpoint, headers, prompt or secret is
accepted by the event API. Missing observations remain null rather than zero.
"""
import math
import os
import threading
import time


def _number(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value < 0):
        raise ValueError('measurement must be non-negative and finite')
    return value


def process_ram_bytes():
    """Current-process RSS via optional psutil; unavailable is unknown."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except (ImportError, OSError, RuntimeError):
        return None


class Performance:
    def __init__(self, sampler=None, interval=0.2, enabled=True):
        self.enabled = enabled
        self.started = time.monotonic()
        self.stages = []
        self.model_lifecycle = []
        self.requests = []
        self.reuse_count = None
        self.resources = {'ram_peak_bytes': None, 'gpu_peak_bytes': None,
                          'sample_count': 0, 'sampling_method': 'not sampled'}
        self.resources.update(cpu_process_seconds=None, cpu_percent_single_core=None,
                              cpu_interval_seconds=None, cpu_sample_count=0,
                              cpu_failure_count=0,
                              cpu_scope='current process, excludes child processes; 100% = one core',
                              cpu_source='time.process_time and time.monotonic',
                              cpu_missing_reason='fewer than two valid CPU samples')
        self._cpu_previous = None
        self._sampler = sampler
        self._interval = _number(interval)
        if self._interval is None or self._interval == 0:
            raise ValueError('sample interval must be positive')
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._active = 0
        self._peak_active = None
        self._sequence = 0
        self._inflight = {}

    def start_sampling(self):
        if not self.enabled:
            return
        if self._sampler is None or self._thread is not None:
            raise ValueError('sampler missing or already started')
        self.resources['sampling_method'] = 'caller-provided RSS/GPU bytes; periodic process sample'
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        while not self._stop.is_set():
            try:
                sample = self._sampler()
                ram, gpu = _number(sample.get('ram_bytes')), _number(sample.get('gpu_bytes'))
                with self._lock:
                    for key, value in (('ram_peak_bytes', ram), ('gpu_peak_bytes', gpu)):
                        if value is not None:
                            prior = self.resources[key]
                            self.resources[key] = value if prior is None else max(value, prior)
                    self.resources['sample_count'] += 1
                    self._record_cpu(sample)
            except Exception:
                with self._lock:
                    self.resources['cpu_failure_count'] += 1
                    self._cpu_previous = None
                    self.resources['cpu_percent_single_core'] = None
                    self.resources['cpu_interval_seconds'] = None
                    self.resources['cpu_missing_reason'] = 'sampling failed; interval unavailable'
            self._stop.wait(self._interval)

    def _record_cpu(self, sample):
        cpu = _number(sample.get('cpu_process_seconds'))
        now = _number(sample.get('monotonic_seconds'))
        if cpu is None or now is None:
            self._cpu_previous = None
            self.resources['cpu_percent_single_core'] = None
            self.resources['cpu_interval_seconds'] = None
            self.resources['cpu_missing_reason'] = 'CPU sample unavailable'
            return
        self.resources['cpu_process_seconds'] = cpu
        self.resources['cpu_sample_count'] += 1
        if self._cpu_previous is not None:
            old_time, old_cpu = self._cpu_previous
            elapsed, delta = now - old_time, cpu - old_cpu
            if elapsed > 0 and delta >= 0:
                self.resources['cpu_interval_seconds'] = elapsed
                self.resources['cpu_percent_single_core'] = 100.0 * delta / elapsed
                self.resources['cpu_missing_reason'] = None
            else:
                self.resources['cpu_interval_seconds'] = None
                self.resources['cpu_percent_single_core'] = None
                self.resources['cpu_missing_reason'] = 'invalid CPU sampling interval'
        self._cpu_previous = (now, cpu)


    def stop_sampling(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def stage(self, name, seconds):
        if not self.enabled:
            return
        _number(seconds)
        with self._lock:
            self.stages.append({'name': name, 'wall_seconds': seconds})

    def timed_stage(self, name, started_unix_s, finished_unix_s, seconds):
        if not self.enabled:
            return
        _number(seconds)
        with self._lock:
            self.stages.append({'name': name, 'started_unix_s': started_unix_s,
                                'finished_unix_s': finished_unix_s, 'wall_seconds': seconds})

    def model_event(self, action, seconds):
        if action not in ('load', 'release'):
            raise ValueError('unknown lifecycle action')
        _number(seconds)
        with self._lock:
            self.model_lifecycle.append({'action': action, 'wall_seconds': seconds})

    def request_start(self, retry=False):
        if type(retry) is not bool:
            raise ValueError('retry must be boolean')
        with self._lock:
            self._sequence += 1
            token = self._sequence
            self._inflight[token] = (time.monotonic(), retry)
            self._active += 1
            self._peak_active = max(self._peak_active or 0, self._active)
            return token

    def request_end(self, token, success, usage=None):
        if type(success) is not bool:
            raise ValueError('success must be boolean')
        if usage is not None:
            if not isinstance(usage, dict) or set(usage) - {'input_tokens', 'output_tokens', 'cached_tokens', 'reasoning_tokens'}:
                raise ValueError('unsupported usage fields')
            usage = {k: _number(v) for k, v in usage.items()}
        with self._lock:
            if token not in self._inflight:
                raise ValueError('unknown or finished request')
            started, retry = self._inflight.pop(token)
            self._active -= 1
            self.requests.append({'wall_seconds': time.monotonic() - started,
                                  'retry': retry, 'success': success, 'usage': usage})

    def reused(self):
        with self._lock:
            self.reuse_count = (self.reuse_count or 0) + 1

    def report(self):
        with self._lock:
            requests = list(self.requests)
            fields = ('input_tokens', 'output_tokens', 'cached_tokens', 'reasoning_tokens')
            totals = {key: (sum(r['usage'][key] for r in requests
                                if r['usage'] is not None and r['usage'].get(key) is not None)
                            if any(r['usage'] is not None and r['usage'].get(key) is not None
                                   for r in requests) else None) for key in fields}
            missing = {key: sum(1 for r in requests if r['usage'] is None or r['usage'].get(key) is None)
                       for key in fields}
            return {'wall_seconds': time.monotonic() - self.started, 'stages': list(self.stages),
                    'model_lifecycle': list(self.model_lifecycle) if self.model_lifecycle else None,
                    'requests': requests if requests else None,
                    'request_count': len(requests) if requests else None,
                    'retry_count': sum(r['retry'] for r in requests) if requests else None,
                    'inflight_count': self._active if self._peak_active is not None else None,
                    'peak_concurrency': self._peak_active, 'reuse_count': self.reuse_count,
                    'usage': totals if requests else None,
                    'usage_unavailable_count': missing if requests else None,
                    'resources': dict(self.resources)}


def sample_resources():
    # Do not import torch merely for measurement or initialize CUDA.
    import sys
    gpu = None
    torch = sys.modules.get('torch')
    try:
        if torch is not None and torch.cuda.is_initialized():
            gpu = torch.cuda.memory_allocated()
    except (AttributeError, RuntimeError):
        pass
    try:
        cpu, now = time.process_time(), time.monotonic()
    except Exception:
        cpu, now = None, None
    return {'ram_bytes': process_ram_bytes(), 'gpu_bytes': gpu,
            'cpu_process_seconds': cpu, 'monotonic_seconds': now}


def summarize_calls(root):
    """Summarize individual call files, never prompt/response or provider details."""
    import json
    from pathlib import Path
    root = Path(root)
    # Calls may belong to nested stage outputs. Count each physical path once,
    # not each prompt (two requests may legitimately have identical prompts).
    paths = sorted({p for directory in root.rglob('calls') if directory.is_dir()
                    for p in directory.rglob('*.json')}) if root.is_dir() else []
    result = {'record_files': len(paths), 'unreadable': 0, 'network_requests': 0,
              'success': 0, 'failure': 0, 'status_unknown': 0, 'reuse': 0,
              'network_unknown': 0, 'duration_known': 0, 'duration_seconds': None,
              'usage_known': 0, 'usage': {'prompt_tokens': None, 'completion_tokens': None},
              'retry_count': None, 'retry_coverage': 'unknown; SDK retries not observable',
              'interval_known': 0, 'observed_peak_concurrency': None,
              'concurrency_scope': 'recorded wrapper intervals, not SDK HTTP attempts'}
    events = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                raise ValueError('not an object')
        except (OSError, ValueError):
            result['unreadable'] += 1
            continue
        if record.get('reused') is True:
            result['reuse'] += 1
            continue
        if record.get('network') is not True:
            result['network_unknown'] += 1
            continue
        result['network_requests'] += 1
        start, end = record.get('request_started_unix_s'), record.get('request_finished_unix_s')
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (start, end)) and end > start >= 0:
            result['interval_known'] += 1
            events.extend(((start, 1), (end, -1)))
        ok = record.get('ok')
        result['success' if ok is True else 'failure' if ok is False else 'status_unknown'] += 1
        duration = record.get('request_duration_s')
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration >= 0:
            result['duration_known'] += 1
            result['duration_seconds'] = (result['duration_seconds'] or 0) + duration
        usage = record.get('usage')
        if record.get('usage_known') is True and isinstance(usage, dict):
            result['usage_known'] += 1
            for key in result['usage']:
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    result['usage'][key] = (result['usage'][key] or 0) + value
    if events:
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        result['observed_peak_concurrency'] = peak
    return result
