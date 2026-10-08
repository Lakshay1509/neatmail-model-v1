"""Private JSONL cost ledger. Only allowlisted, content-free fields are accepted."""
import contextvars
import json
import math
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

FIELDS = set('experiment config request_id item_id item_ids cohort variant entry_point status '
             'path fallback_stage category_kind response_required summary_needed elapsed_ms '
             'stage attempt provider model usage provider_id error_code item_count batch_size '
             'answers telemetry_healthy retry_visibility'.split())
PRICING_VERSION = '2026-10-08'


def safe_identifier(value):
    """Only retain bounded provider identifiers, never arbitrary response objects."""
    import re
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_./:-]{1,160}', value) else None


def normalize_usage(usage, provider, model):
    usage = usage or {}
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        usage = {}
    inp = usage.get('input_tokens', usage.get('prompt_tokens'))
    out = usage.get('output_tokens', usage.get('completion_tokens'))
    details = usage.get('prompt_tokens_details') or {}
    if not isinstance(details, dict):
        details = {}
    cached = details.get('cached_tokens', 0)
    output_details = usage.get('completion_tokens_details') or {}
    reasoning = output_details.get('reasoning_tokens', 0) if isinstance(output_details, dict) else None
    cost, source = None, 'unknown'
    valid = lambda n: type(n) in (int, float) and math.isfinite(n) and n >= 0
    if provider == 'openrouter' and valid(usage.get('cost')):
        cost, source = usage['cost'], 'reported'
    elif provider == 'openai' and valid(inp):
        if model.startswith('gpt-5-nano') and valid(out) and valid(cached) and cached <= inp:
            cost = ((inp - cached) * .05 + cached * .005 + out * .40) / 1e6
        elif model == 'text-embedding-3-small':
            cost = inp * .02 / 1e6
        if cost is not None:
            source = 'estimated'
    return dict(input_tokens=inp if valid(inp) else None, output_tokens=out if valid(out) else None,
                cached_tokens=cached if valid(cached) else None,
                reasoning_tokens=reasoning if valid(reasoning) else None, cost=cost, cost_source=source,
                pricing_version=PRICING_VERSION)


class EventWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.boot = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.healthy = True
        probe = self.directory / f'.probe-{self.boot}'
        probe.write_text('', encoding='utf-8')
        probe.unlink()

    def emit(self, event, **fields):
        if fields.keys() - FIELDS:
            raise ValueError('Telemetry contains fields outside the privacy allowlist')
        now = datetime.now(timezone.utc)
        record = dict(schema_version=1, event=event, event_id=uuid.uuid4().hex,
                      timestamp=now.isoformat(), **fields)
        line = json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n'
        try:
            with self.lock:
                path = self.directory / f'{now:%Y-%m-%d}-{self.boot}.jsonl'
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(line)
                    stream.flush()
        except OSError:
            self.healthy = False
            print('{"event":"ab_telemetry_write_failed"}', file=sys.stderr)


CURRENT = contextvars.ContextVar('ab_observer', default=None)


@contextmanager
def observe(observer):
    token = CURRENT.set(observer)
    try:
        yield
    finally:
        CURRENT.reset(token)


def observed_call(stage, provider, model, call):
    """Keep legacy SDK retry behavior; disclose that inner retries are opaque."""
    observer = CURRENT.get()
    started = time.monotonic()
    try:
        response = call()
    except Exception as exc:
        if observer:
            observer(stage=stage, provider=provider, model=model, usage=None,
                     elapsed_ms=(time.monotonic() - started) * 1000,
                     status='error', error_code=str(getattr(exc, 'status_code', 'network')),
                     retry_visibility='sdk_internal')
        raise
    if observer:
        observer(stage=stage, provider=provider, model=safe_identifier(getattr(response, 'model', model)) or model,
                 usage=normalize_usage(getattr(response, 'usage', None), provider, model),
                 elapsed_ms=(time.monotonic() - started) * 1000, status='ok',
                 provider_id=safe_identifier(getattr(response, 'id', None)), retry_visibility='sdk_internal')
    return response
