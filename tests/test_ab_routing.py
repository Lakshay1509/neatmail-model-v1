import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from classification_ab import Settings, TreatmentError, assign_variant
from ab_routing import Experiment
from test_classification_ab import email


class FakeTreatment:
    def __init__(self, fail=False):
        self.fail = fail

    def classify(self, item, corrections, guidance, observer):
        if self.fail:
            raise TreatmentError('summary', '401')
        return dict(category='Updates', response_required=False, ai_summary='', ai_action='')

    def close(self):
        pass


class RoutingTests(unittest.TestCase):
    def make_experiment(self, folder, percent=50, treatment=None):
        return Experiment(Settings(enabled=True, percent=percent, key='mock', cohort_key='test',
                                   data_dir=folder, report_dir=folder), treatment=treatment or FakeTreatment())

    def control(self, item, corrections):
        self.prepared.append(corrections)
        return dict(category='Pending Response', response_required=True, ai_summary='', ai_action='')

    def test_fallback_reuses_corrections_and_retains_assignment(self):
        self.prepared = []
        with tempfile.TemporaryDirectory() as folder:
            experiment = self.make_experiment(folder, 100, FakeTreatment(True))
            retrieved = []
            def retrieve(item):
                retrieved.append(item)
                return [{'correct_label': 'Updates'}]
            result = experiment.single(email(), retrieve, self.control, lambda s: s)
            experiment.close()
            self.assertEqual(result['category'], 'Pending Response')
            self.assertEqual(len(retrieved), 1)
            self.assertEqual(self.prepared, [[{'correct_label': 'Updates'}]])
            events = [json.loads(line) for p in Path(folder).glob('*.jsonl') for line in p.read_text().splitlines()]
            end = next(e for e in events if e['event'] == 'item_end')
            self.assertEqual((end['variant'], end['path'], end['fallback_stage']),
                             ('treatment', 'control_fallback', 'summary'))
            self.assertNotIn('private-user', json.dumps(events))

    def test_mixed_batch_order_duplicate_ids_and_control_grouping(self):
        self.prepared = []
        with tempfile.TemporaryDirectory() as folder:
            experiment = self.make_experiment(folder)
            items = []
            for variant in ['treatment', 'control', 'treatment', 'control']:
                item = email()
                item.user_id = next(str(i) for i in range(100) if assign_variant(str(i), experiment.settings.experiment, 50) == variant)
                item.id = 'duplicate'
                items.append(item)
            batches = []
            def control_batch(rows, corrections):
                batches.append(rows)
                return [dict(id=row.id, category='Pending Response', response_required=True,
                             ai_summary='', ai_action='') for row in reversed(rows)]
            result = experiment.batch(items, lambda item: [], self.control, control_batch, lambda s: s)
            experiment.close()
            self.assertEqual([row['category'] for row in result],
                             ['Updates', 'Pending Response', 'Updates', 'Pending Response'])
            self.assertEqual([row['id'] for row in result], ['duplicate'] * 4)
            self.assertEqual(len(batches), 1)
            self.assertEqual(len({r.id for r in batches[0]}), 2)


if __name__ == '__main__':
    unittest.main()
