import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from classification_ab import Settings, Treatment, TreatmentError


def email(tags=None, sender='person@example.com'):
    return SimpleNamespace(user_id='private-user', subject='Question', from_=sender,
                           bodySnippet='Please approve the design.', sensitivity='if actionable',
                           tags=tags if tags is not None else [
                               SimpleNamespace(name='Pending Response', description='', user_defined=False),
                               SimpleNamespace(name='Action Needed', description='', user_defined=False),
                               SimpleNamespace(name='Updates', description='', user_defined=True)])


def answers(payload, category='topic', automated=0, cold=0, obligation=1, judgment=1, reply=1):
    result = {}
    for key, q in payload['questions'].items():
        if q['type'] == 'choice':
            choice = 'none'
            if key == 'topic':
                choice = 't2'
            result[key] = {'type': 'choice', 'choice': choice, 'confidence': 1,
                           'probabilities': {k: float(k == choice) for k in q['criteria']}}
        else:
            value = dict(automated=automated, cold=cold, obligation=obligation,
                         judgment=judgment, reply=reply)[key]
            result[key] = {'type': 'noul', 'noul': value}
    return {'model': 'cloudflare/clef-flash', 'answers': result,
            'usage': {'input_tokens': 20, 'output_tokens': 5, 'cost': .001}}


class TreatmentTests(unittest.TestCase):
    def test_transient_credit_hold_retries_but_empty_balance_does_not(self):
        for transient in (True, False):
            calls, events = [], []
            def respond(request):
                calls.append(request)
                if len(calls) == 1:
                    return httpx.Response(402, headers={'Retry-After': '0'}, json={'error': {
                        'metadata': {'limit_source': 'openrouter_in_flight_budget' if transient else 'openrouter_credits'}}})
                return httpx.Response(200, json={'usage': {'cost': .01}})
            treatment = Treatment(Settings(key='mock'), httpx.Client(transport=httpx.MockTransport(respond)))
            import time
            try:
                with patch('classification_ab.time.sleep'):
                    if transient:
                        treatment._post('classifier', 'https://openrouter.ai/api/alpha/decisions',
                                        {'model': 'cloudflare/clef-flash'}, time.monotonic()+60,
                                        lambda **e: events.append(e))
                        self.assertEqual(len(calls), 2)
                    else:
                        with self.assertRaises(TreatmentError):
                            treatment._post('classifier', 'https://openrouter.ai/api/alpha/decisions',
                                            {'model': 'cloudflare/clef-flash'}, time.monotonic()+60,
                                            lambda **e: events.append(e))
                        self.assertEqual(len(calls), 1)
            finally:
                treatment.close()

    def run_case(self, item=None, transform=None, values=None, summary_status=200, cfg=None):
        self.calls, self.events = [], []
        def respond(request):
            payload = json.loads(request.content)
            self.calls.append((str(request.url), payload))
            if 'decisions' in str(request.url):
                response = answers(payload, **(values or {}))
                if transform:
                    transform(response)
                return httpx.Response(200, json=response)
            return httpx.Response(summary_status, json={
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                    'ai_summary': 'Approve the proposed design so the team can complete the next project milestone.',
                    'ai_action': 'Review & approve'})}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'cost': .002}})
        client = httpx.Client(transport=httpx.MockTransport(respond))
        treatment = Treatment(cfg or Settings(key='mock'), http_client=client)
        self.result = treatment.classify(item or email(), [], 'Apply sensitivity.',
                                         lambda **event: self.events.append(event))
        return self.result

    def test_actionable_calls_summary_on_openrouter(self):
        result = self.run_case()
        self.assertEqual(result['category'], 'Pending Response')
        self.assertTrue(result['response_required'])
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(url.startswith('https://openrouter.ai/') for url, _ in self.calls))
        self.assertNotIn('private-user', json.dumps(self.calls))
        self.assertEqual(self.calls[1][1]['provider'], {'require_parameters': True})

    def test_clef_decision_model_and_optional_confidence(self):
        def omit_confidence(response):
            for answer in response['answers'].values():
                answer.pop('confidence', None)
        result = self.run_case(transform=omit_confidence)
        self.assertEqual(result['category'], 'Pending Response')
        self.assertEqual(self.calls[0][1]['model'], 'cloudflare/clef-flash')
        self.assertEqual(self.calls[0][1]['provider'],
                         {'only': ['primeintellect'], 'allow_fallbacks': False})

    def test_invalid_optional_confidence_falls_back(self):
        with self.assertRaises(TreatmentError):
            self.run_case(transform=lambda r: r['answers']['topic'].update(confidence=2))

    def test_topic_cold_and_digest_skip_summary(self):
        for item, values in [(email(), {'obligation': 0, 'reply': 0}),
                             (email(), {'cold': 1}),
                             (email(sender='digest@send.neatmail.app'), {})]:
            result = self.run_case(item=item, values=values)
            self.assertEqual(result['ai_summary'], '')
            self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.run_case(values={'cold': 1})['response_required'])

    def test_automated_decision_and_one_click(self):
        result = self.run_case(values={'automated': 1})
        self.assertEqual(result['category'], 'Action Needed')
        self.assertFalse(result['response_required'])
        result = self.run_case(values={'automated': 1, 'judgment': 0})
        self.assertEqual(result['ai_summary'], '')

    def test_empty_tags_skip_providers(self):
        result = self.run_case(email(tags=[]))
        self.assertEqual(result['category'], '')
        self.assertEqual(self.calls, [])

    def test_invalid_probability_captures_cost_before_failure(self):
        with self.assertRaises(TreatmentError):
            self.run_case(transform=lambda r: r['answers']['cold'].update(noul=2))
        self.assertEqual(self.events[0]['usage']['cost'], .001)

    def test_summary_auth_failure_not_retried(self):
        with self.assertRaises(TreatmentError) as raised:
            self.run_case(summary_status=401)
        self.assertEqual(raised.exception.stage, 'summary')
        self.assertEqual(len(self.calls), 2)

    def test_correction_precedence_and_private_evidence(self):
        events = []
        def respond(request):
            payload = json.loads(request.content)
            data = answers(payload, obligation=0, reply=0, cold=1)
            correction = data['answers']['correction']
            correction.update(choice='c0', probabilities={'c0': 1., 'none': 0.}, leaked='private body')
            return httpx.Response(200, json=data)
        treatment = Treatment(Settings(key='mock'), http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        result = treatment.classify(email(), [{'correct_label': 'Updates', 'snippet': 'private body'}],
                                    'standard', lambda **event: events.append(event))
        self.assertEqual(result['category'], 'Updates')
        self.assertNotIn('private body', json.dumps(events))

    def test_ambiguous_topic_abstains(self):
        def uncertain(response):
            response['answers']['topic'].update(probabilities={'t0': .1, 't1': .1, 't2': .7, 'none': .1}, confidence=.6)
        result = self.run_case(values={'obligation': 0, 'reply': 0}, transform=uncertain)
        self.assertEqual(result['category'], '')
        self.assertEqual(len(self.calls), 1)

    def test_retry_transient_and_record_both_attempts(self):
        calls, events = [], []
        def respond(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={'retry-after': '0'})
            return httpx.Response(200, json=answers(json.loads(request.content), obligation=0, reply=0))
        treatment = Treatment(Settings(key='mock'), http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        with patch('classification_ab.time.sleep'):
            result = treatment.classify(email(), [], 'standard', lambda **event: events.append(event))
        self.assertEqual(result['category'], 'Updates')
        self.assertEqual([e['attempt'] for e in events if e['stage'] == 'classifier'], [1, 2])

    def test_unknown_choice_and_too_many_tags_fail(self):
        with self.assertRaises(TreatmentError):
            self.run_case(transform=lambda r: r['answers']['topic'].update(choice='secret'))
        with self.assertRaises(TreatmentError):
            self.run_case(email(tags=email().tags * 85))


if __name__ == '__main__':
    unittest.main()
