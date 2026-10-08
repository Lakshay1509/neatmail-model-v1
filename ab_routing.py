"""Single/batch orchestration and attribution, separate from model policy."""
import copy
import hashlib
import hmac
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore

from ab_metrics import EventWriter, observe
from classification_ab import Treatment, TreatmentError, assign_variant, category_kind


def as_dict(result):
    return result.model_dump() if hasattr(result, 'model_dump') else dict(result)


class Experiment:
    def __init__(self, settings, treatment=None):
        self.settings = settings
        self.config = settings.fingerprint()
        self.writer = EventWriter(settings.data_dir)
        Path(settings.report_dir).mkdir(parents=True, exist_ok=True)
        probe = Path(settings.report_dir) / f'.probe-{uuid.uuid4().hex}'
        probe.write_text('', encoding='utf-8')
        probe.unlink()
        self.treatment = treatment or (Treatment(settings) if settings.percent else None)
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='classification-ab')
        self.slots = BoundedSemaphore(16)

    def close(self):
        self.pool.shutdown(wait=True)
        if self.treatment:
            self.treatment.close()

    def _submit(self, call, *args):
        self.slots.acquire()
        def run():
            try:
                return call(*args)
            finally:
                self.slots.release()
        try:
            return self.pool.submit(run)
        except Exception:
            self.slots.release()
            raise

    def _emit(self, event, **fields):
        self.writer.emit(event, experiment=self.settings.experiment, config=self.config, **fields)

    def _item(self, email, request_id, entry_point):
        variant = assign_variant(email.user_id, self.settings.experiment, self.settings.percent)
        cohort = hmac.new(self.settings.cohort_key.encode(), email.user_id.encode(), hashlib.sha256).hexdigest()
        ctx = dict(request_id=request_id, item_id=uuid.uuid4().hex, cohort=cohort,
                   variant=variant, entry_point=entry_point)
        self._emit('item_start', **ctx)
        return ctx

    def _observer(self, contexts, summary_counts):
        def record(**fields):
            if fields['stage'] == 'evidence':
                for ctx in contexts:
                    self._emit('evidence', **ctx, **fields)
                return
            if fields['stage'] == 'summary':
                for ctx in contexts:
                    summary_counts[ctx['item_id']] += 1
            self._emit('provider_attempt', request_id=contexts[0]['request_id'],
                       item_ids=[ctx['item_id'] for ctx in contexts],
                       variant=contexts[0]['variant'], entry_point=contexts[0]['entry_point'],
                       batch_size=len(contexts), **fields)
        return record

    def _finish(self, ctx, started, counts, result=None, path=None, fallback=None):
        fields = dict(status='ok' if result is not None else 'error', path=path or ctx['variant'],
                      fallback_stage=fallback, elapsed_ms=(time.monotonic() - started) * 1000,
                      summary_needed=counts[ctx['item_id']], telemetry_healthy=self.writer.healthy)
        if result is not None:
            fields.update(category_kind=category_kind(result['category']),
                          response_required=result['response_required'])
        self._emit('item_end', **ctx, **fields)

    def _one(self, email, ctx, started, counts, retrieve, control, sensitivity):
        result, path, fallback = None, ctx['variant'], None
        observer = self._observer([ctx], counts)
        try:
            with observe(observer):
                corrections = retrieve(email)
                if ctx['variant'] == 'treatment':
                    try:
                        result = self.treatment.classify(email, corrections, sensitivity(email.sensitivity), observer)
                    except TreatmentError as exc:
                        path, fallback = 'control_fallback', exc.stage
                        result = as_dict(control(email, corrections))
                else:
                    result = as_dict(control(email, corrections))
            return result
        finally:
            self._finish(ctx, started, counts, result, path, fallback)

    def single(self, email, retrieve, control, sensitivity, trace_callback=None):
        started, request_id = time.monotonic(), uuid.uuid4().hex
        self._emit('request_start', request_id=request_id, entry_point='classify', item_count=1)
        ctx = self._item(email, request_id, 'classify')
        if trace_callback:
            trace_callback([ctx['item_id']])
        counts = {ctx['item_id']: 0}
        status = 'error'
        try:
            result = self._submit(self._one, email, ctx, started, counts, retrieve, control, sensitivity).result()
            status = 'ok'
            return result
        finally:
            self._emit('request_end', request_id=request_id, entry_point='classify', status=status,
                       item_count=1, elapsed_ms=(time.monotonic() - started) * 1000,
                       telemetry_healthy=self.writer.healthy)

    def _control_batch(self, rows, contexts, started, counts, retrieve, control_batch):
        results = None
        try:
            observer = self._observer(contexts, counts)
            with observe(observer):
                # Embeddings are attributed to their individual items, batch completion once.
                corrections = []
                for row, ctx in zip(rows, contexts):
                    with observe(self._observer([ctx], counts)):
                        corrections.append(retrieve(row))
                internal = []
                for row, ctx in zip(rows, contexts):
                    clone = copy.copy(row)
                    clone.id = ctx['item_id']
                    internal.append(clone)
                raw = [as_dict(r) for r in control_batch(internal, corrections)]
                if len(raw) != len(rows) or {r['id'] for r in raw} != {r.id for r in internal}:
                    raise ValueError('Invalid control batch IDs')
                lookup = {r['id']: r for r in raw}
                results = [dict(lookup[ctx['item_id']], id=row.id) for row, ctx in zip(rows, contexts)]
                return results
        finally:
            for i, ctx in enumerate(contexts):
                self._finish(ctx, started, counts, results[i] if results is not None else None)

    def batch(self, rows, retrieve, control, control_batch, sensitivity, trace_callback=None):
        if not rows:
            return []
        started, request_id = time.monotonic(), uuid.uuid4().hex
        self._emit('request_start', request_id=request_id, entry_point='classify_batch', item_count=len(rows))
        contexts = [self._item(row, request_id, 'classify_batch') for row in rows]
        if trace_callback:
            trace_callback([ctx['item_id'] for ctx in contexts])
        counts = {ctx['item_id']: 0 for ctx in contexts}
        control_indices = [i for i, ctx in enumerate(contexts) if ctx['variant'] == 'control']
        futures = []
        if control_indices:
            futures.append((control_indices, self._submit(self._control_batch,
                [rows[i] for i in control_indices], [contexts[i] for i in control_indices],
                started, counts, retrieve, control_batch)))
        for i, ctx in enumerate(contexts):
            if ctx['variant'] == 'treatment':
                futures.append(([i], self._submit(self._one, rows[i], ctx, started, counts,
                                                 retrieve, control, sensitivity)))
        result, error = [None] * len(rows), None
        try:
            # Await every submitted task, even if another fails; no orphan billable work.
            for indices, future in futures:
                try:
                    values = future.result()
                    if len(indices) == 1 and isinstance(values, dict):
                        values = [values]
                    for i, value in zip(indices, values):
                        result[i] = dict(value, id=rows[i].id)
                except Exception as exc:
                    error = error or exc
            if error:
                raise error
            return result
        finally:
            self._emit('request_end', request_id=request_id, entry_point='classify_batch',
                       status='error' if error else 'ok', item_count=len(rows),
                       elapsed_ms=(time.monotonic() - started) * 1000,
                       telemetry_healthy=self.writer.healthy)
