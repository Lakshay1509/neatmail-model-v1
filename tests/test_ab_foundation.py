import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

from classification_ab import Settings, assign_variant
from ab_metrics import EventWriter, normalize_usage


class FoundationTests(unittest.TestCase):
    def test_assignment_stable_and_boundaries(self):
        for i in range(100):
            user = f'user-{i}'
            self.assertEqual(assign_variant(user, 'v1', 0), 'control')
            self.assertEqual(assign_variant(user, 'v1', 100), 'treatment')
            self.assertEqual(assign_variant(user, 'v1', 50), assign_variant(user, 'v1', 50))
        counts = sum(assign_variant(str(i), 'v1', 50) == 'treatment' for i in range(1000))
        self.assertTrue(400 < counts < 600)

    def test_config_safe_defaults_and_zero_percent_rollback(self):
        self.assertFalse(Settings.from_env({}).enabled)
        cfg = Settings.from_env({'CLASSIFICATION_AB_ENABLED': 'true',
                                 'CLASSIFICATION_AB_TREATMENT_PERCENT': '0',
                                 'AB_COHORT_HMAC_KEY': 'test'})
        self.assertEqual(cfg.percent, 0)
        with self.assertRaises(ValueError):
            Settings.from_env({'CLASSIFICATION_AB_ENABLED': 'true'})
        for invalid in ('nan', '-1', '101'):
            with self.assertRaises(ValueError):
                Settings.from_env({'CLASSIFICATION_AB_TREATMENT_PERCENT': invalid})

    def test_clef_defaults_and_legacy_model_setting_rejected(self):
        cfg = Settings.from_env({})
        self.assertEqual(cfg.decision_model, 'cloudflare/clef-flash')
        self.assertEqual(cfg.experiment, 'clef-flash-v1')
        self.assertEqual(cfg.decision_provider, 'primeintellect')
        other = Settings.from_env({'OPENROUTER_DECISION_PROVIDER': 'cloudflare'})
        self.assertNotEqual(cfg.fingerprint(), other.fingerprint())
        with self.assertRaises(ValueError):
            Settings.from_env({'OPENROUTER_DECISION_PROVIDER': 'unknown-provider'})
        with self.assertRaises(ValueError):
            Settings.from_env({'OPENROUTER_JEV_MODEL': 'typesafe/jev-1.13'})

    def test_private_writer_and_restart_files(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = EventWriter(folder), EventWriter(folder)
            a.emit('item_start', request_id='r', item_id='i', variant='control')
            b.emit('item_end', request_id='r', item_id='i', status='ok')
            files = list(Path(folder).glob('*.jsonl'))
            self.assertEqual(len(files), 2)
            events = [json.loads(p.read_text()) for p in files]
            self.assertEqual(len({e['event_id'] for e in events}), 2)
            with self.assertRaises(ValueError):
                a.emit('item_start', body='sensitive')

    def test_cost_does_not_double_count_reasoning(self):
        usage = {'prompt_tokens': 1000, 'prompt_tokens_details': {'cached_tokens': 500},
                 'completion_tokens': 100, 'completion_tokens_details': {'reasoning_tokens': 90}}
        got = normalize_usage(usage, 'openai', 'gpt-5-nano')
        self.assertAlmostEqual(got['cost'], (500 * .05 + 500 * .005 + 100 * .4) / 1e6)
        self.assertEqual(got['cost_source'], 'estimated')
        self.assertIsNone(normalize_usage(None, 'openai', 'gpt-5-nano')['cost'])
        self.assertEqual(normalize_usage({'input_tokens': 5, 'output_tokens': 1, 'cost': 0},
                                        'openrouter', 'cloudflare/clef-flash')['cost'], 0)

    def test_embedding_cost_without_completion_tokens(self):
        self.assertAlmostEqual(normalize_usage({'prompt_tokens': 1000}, 'openai',
                                              'text-embedding-3-small')['cost'], .00002)

    def test_assignment_across_processes(self):
        output = subprocess.check_output([sys.executable, '-c',
            "from classification_ab import assign_variant; print(assign_variant('user', 'v1', 50))"], text=True)
        self.assertEqual(output.strip(), assign_variant('user', 'v1', 50))

    def test_unavailable_storage_and_runtime_write_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            blocked = Path(folder, 'file')
            blocked.write_text('not a directory')
            with self.assertRaises(OSError):
                EventWriter(blocked)
            writer = EventWriter(folder)
            writer.directory = blocked
            writer.emit('item_start', item_id='test')
            self.assertFalse(writer.healthy)


if __name__ == '__main__':
    unittest.main()
