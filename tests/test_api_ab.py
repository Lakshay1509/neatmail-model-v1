import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from test_ab_routing import FakeTreatment


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLASSIFICATION_AB_ENABLED': 'true',
            'CLASSIFICATION_AB_TREATMENT_PERCENT': '100', 'OPENROUTER_API_KEY': 'mock',
            'AB_COHORT_HMAC_KEY': 'mock', 'AB_DATA_DIR': self.folder.name,
            'AB_REPORT_DIR': self.folder.name, 'OPENAI_API_KEY': 'mock',
            'PINECONE_API_KEY': 'mock', 'DASHBOARD_API_KEY': 'test-key'})
        self.env.start()
        pc = MagicMock()
        pc.list_indexes.return_value = [SimpleNamespace(name='neatmail-corrections')]
        self.openai = MagicMock()
        sys.modules.pop('main', None)
        with patch('pinecone.Pinecone', return_value=pc), patch('openai.OpenAI', return_value=self.openai), \
                patch('dotenv.load_dotenv'):
            self.app_module = importlib.import_module('main')
        self.app_module.experiment.treatment.close()
        self.app_module.experiment.treatment = FakeTreatment()
        self.app_module.get_corrections = MagicMock(return_value=[])
        self.api = TestClient(self.app_module.app)

    def tearDown(self):
        if self.app_module.experiment:
            self.app_module.experiment.close()
        self.env.stop()
        self.folder.cleanup()
        sys.modules.pop('main', None)

    def payload(self):
        return dict(user_id='private-user', subject='private-subject', **{'from': 'private@example.com'},
                    bodySnippet='private body', tags=[dict(name='Updates')], sensitivity='if actionable')

    def test_single_auth_contract_and_trace_header(self):
        self.assertEqual(self.api.post('/classify', json=self.payload()).status_code, 401)
        response = self.api.post('/classify', json=self.payload(), headers={'X-API-Key': 'test-key'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {'category', 'response_required', 'ai_summary', 'ai_action'})
        self.assertEqual(len(response.headers['X-AB-Item-IDs']), 32)
        self.openai.chat.completions.create.assert_not_called()
        logs = ''.join(p.read_text() for p in Path(self.folder.name).glob('*.jsonl'))
        for secret in ('private-user', 'private-subject', 'private body', 'private@example.com'):
            self.assertNotIn(secret, logs)

    def test_batch_order_ids_and_limit(self):
        rows = [dict(self.payload(), id='duplicate') for _ in range(3)]
        response = self.api.post('/classify-batch', json={'requests': rows}, headers={'X-API-Key': 'test-key'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([r['id'] for r in response.json()['results']], ['duplicate']*3)
        self.assertEqual(len(response.headers['X-AB-Item-IDs'].split(',')), 3)
        self.assertEqual(self.api.post('/classify-batch', json={'requests': rows*4},
                                      headers={'X-API-Key': 'test-key'}).status_code, 400)

    def test_disabled_control_keeps_original_parameters(self):
        self.app_module.experiment.close()
        self.app_module.experiment = None
        self.openai.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                'category': 'Updates', 'response_required': False, 'ai_summary': '', 'ai_action': ''})))])
        response = self.api.post('/classify', json=self.payload(), headers={'X-API-Key': 'test-key'})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('X-AB-Item-IDs', response.headers)
        kwargs = self.openai.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs['model'], 'gpt-5-nano')
        self.assertEqual(kwargs['max_completion_tokens'], 6000)
        self.assertEqual(kwargs['reasoning_effort'], 'medium')

    def test_provider_error_does_not_expose_email_or_secret(self):
        self.app_module.experiment.close()
        self.app_module.experiment = None
        self.openai.chat.completions.create.side_effect = RuntimeError('private body secret-api-key')
        response = self.api.post('/classify', json=self.payload(), headers={'X-API-Key': 'test-key'})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('private body', response.text)
        self.assertNotIn('secret-api-key', response.text)


if __name__ == '__main__':
    unittest.main()
