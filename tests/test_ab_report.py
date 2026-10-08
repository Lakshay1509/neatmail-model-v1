import json
import tempfile
import unittest
from pathlib import Path

from eval.report_ab import build_report, write_report


def event(name, ident, **fields):
    return dict(schema_version=1, event=name, event_id=ident, timestamp='2026-10-09T10:00:00+00:00',
                experiment='clef-flash-v1', config='v1', **fields)


class ReportTests(unittest.TestCase):
    def test_incomplete_without_calls_is_unknown_and_damaged_events_are_skipped(self):
        rows = [event('item_start', 'a', item_id='a', request_id='r', cohort='x',
                      variant='treatment', entry_point='classify'),
                event('provider_attempt', 'bad', item_ids=[], stage='classifier', elapsed_ms=1),
                {'schema_version': 1, 'event_id': 'missing', 'timestamp': '2026-10-09'}]
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'e.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            report = build_report(folder, 'clef-flash-v1', '2026-10-08', '2026-10-15')
            self.assertEqual(report['invalid_lines'], 2)
            self.assertIsNone(report['groups'][0]['inference_cost_per_item'])

    def test_descriptive_treatment_minus_control(self):
        rows = []
        for variant, latency, cost in [('control', 100, .02), ('treatment', 60, .01)]:
            rows += [event('item_start', variant, item_id=variant, request_id=variant, cohort=variant,
                           variant=variant, entry_point='classify'),
                     event('item_end', variant+'e', item_id=variant, status='ok', elapsed_ms=latency),
                     event('provider_attempt', variant+'p', item_ids=[variant], stage='control',
                           elapsed_ms=latency, usage={'cost': cost})]
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'e.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            report = build_report(folder, 'clef-flash-v1', '2026-10-08', '2026-10-15')
            diff = report['comparisons'][0]['treatment_minus_control']
            self.assertAlmostEqual(diff['inference_cost_per_item'], -.01)
            self.assertEqual(diff['item_latency_p50_ms'], -40)
            self.assertIsNone(diff['classification_accuracy'])

    def test_batch_cost_counted_once_and_quality_unknown(self):
        rows = []
        for i in ['a', 'b']:
            rows += [event('item_start', i, item_id=i, request_id='r', cohort=i, variant='control', entry_point='classify_batch'),
                     event('item_end', i+'e', item_id=i, request_id='r', status='ok', category_kind='topic',
                           path='control', elapsed_ms=10, summary_needed=0)]
        rows += [event('provider_attempt', 'cost', item_ids=['a', 'b'], stage='control',
                       usage={'cost': .1}, elapsed_ms=2, retry_visibility='explicit')]
        rows += [rows[0]]  # Duplicate exported line must not duplicate denominators.
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n{truncated', encoding='utf-8')
            report = build_report(folder, 'clef-flash-v1', '2026-10-08', '2026-10-15')
            group = report['groups'][0]
            self.assertEqual(group['attempted_items'], 2)
            self.assertAlmostEqual(group['known_inference_cost'], .1)
            self.assertEqual(group['classification_accuracy'], None)
            self.assertEqual(report['invalid_lines'], 1)
            paths = write_report(report, folder)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertIn('not measured', paths[0].read_text())

    def test_incomplete_missing_cost_and_dates(self):
        rows = [event('item_start', 'a', item_id='a', request_id='r', cohort='x', variant='treatment', entry_point='classify'),
                event('provider_attempt', 'cost', item_ids=['a'], stage='classifier', usage={'cost': None}, elapsed_ms=2)]
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'e.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            report = build_report(folder, 'clef-flash-v1', '2026-10-08', '2026-10-15')
            group = report['groups'][0]
            self.assertEqual(group['incomplete_items'], 1)
            self.assertEqual(group['cost_coverage'], 0)
            self.assertIsNone(group['inference_cost_per_item'])
            self.assertEqual(build_report(folder, 'clef-flash-v1', '2026-10-10', '2026-10-15')['groups'], [])
            with self.assertRaises(ValueError):
                build_report(folder, 'clef-flash-v1', '2026-10-15', '2026-10-08')

    def test_quality_join_and_conflicting_review_rejected(self):
        rows = [event('item_start', 'a', item_id='a', request_id='r', cohort='x', variant='treatment', entry_point='classify'),
                event('item_end', 'b', item_id='a', request_id='r', status='ok', category_kind='topic',
                      path='treatment', elapsed_ms=10, summary_needed=0)]
        review = {'item_id': 'a', 'category_correct': True, 'response_required_correct': False,
                  'summary_correct': None, 'reviewer': 'reviewer-1'}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'e.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            path = Path(folder, 'quality.json')
            path.write_text(json.dumps([review]))
            group = build_report(folder, 'clef-flash-v1', '2026-10-08', '2026-10-15', path)['groups'][0]
            self.assertEqual(group['classification_accuracy'], 1)
            self.assertEqual(group['reply_accuracy'], 0)
            self.assertIsNone(group['summary_accuracy'])
            path.write_text(json.dumps([review, review]))
            with self.assertRaises(ValueError):
                build_report(folder, 'clef-flash-v1', '2026-10-08', '2026-10-15', path)


if __name__ == '__main__':
    unittest.main()
