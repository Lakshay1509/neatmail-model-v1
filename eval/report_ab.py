"""Offline weekly experiment reports; never reads emails or makes provider calls."""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def durations(values):
    return {'p50_ms': percentile(values, .5), 'p95_ms': percentile(values, .95)}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def validate_event(event):
    """Reject damaged exports before they can distort totals or crash aggregation."""
    required = {
        'item_start': ('item_id', 'request_id', 'config', 'entry_point', 'variant', 'cohort'),
        'item_end': ('item_id', 'status'),
        'request_end': ('request_id',),
        'request_start': ('request_id',),
        'evidence': ('item_id',),
        'provider_attempt': ('stage',),
    }
    name = event.get('event')
    if name not in required or any(not isinstance(event.get(k), str) or not event[k] for k in required[name]):
        raise ValueError('Invalid event fields')
    if name in {'item_end', 'request_end', 'provider_attempt'}:
        elapsed = event.get('elapsed_ms')
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid duration')
    if name == 'item_end' and type(event.get('summary_needed', 0)) is not int:
        raise ValueError('Invalid summary count')
    if name == 'provider_attempt':
        ids = event.get('item_ids')
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid provider item IDs')
        if event.get('usage') is not None and not isinstance(event['usage'], dict):
            raise ValueError('Invalid usage')


def quality_records(filename):
    if not filename:
        return {}
    rows = json.loads(Path(filename).read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('Quality file must contain a JSON array')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'item_id', 'category_correct', 'response_required_correct', 'summary_correct', 'reviewer'}:
            raise ValueError('Quality record has invalid fields')
        if not isinstance(row['item_id'], str) or row['item_id'] in result or not row['reviewer']:
            raise ValueError('Quality records need unique item IDs and reviewer provenance')
        for key in ('category_correct', 'response_required_correct'):
            if type(row[key]) is not bool:
                raise ValueError('Quality correctness values must be booleans')
        if row['summary_correct'] is not None and type(row['summary_correct']) is not bool:
            raise ValueError('Summary correctness must be boolean or null')
        result[row['item_id']] = row
    return result


def build_report(directory, experiment, start, end, quality_file=None):
    beginning, ending = utc(start), utc(end)
    if beginning >= ending:
        raise ValueError('Report end must be after start')
    events, seen, invalid, duplicates = [], set(), 0, 0
    root = Path(directory)
    if not root.is_dir():
        raise ValueError('Measurement directory does not exist')
    for path in sorted(root.glob('*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                    timestamp = utc(event['timestamp'])
                    if event.get('schema_version') != 1 or not isinstance(event['event_id'], str):
                        raise ValueError('Unsupported event')
                    validate_event(event)
                    if event['event_id'] in seen:
                        duplicates += 1
                        continue
                    seen.add(event['event_id'])
                    if beginning <= timestamp < ending and event.get('experiment') == experiment:
                        events.append(event)
                except (ValueError, KeyError, TypeError, AttributeError):
                    invalid += 1
    starts = {e['item_id']: e for e in events if e['event'] == 'item_start'}
    ends = {e['item_id']: e for e in events if e['event'] == 'item_end'}
    requests = {e['request_id']: e for e in events if e['event'] == 'request_end'}
    attempts = [e for e in events if e['event'] == 'provider_attempt']
    quality = quality_records(quality_file)
    grouped = defaultdict(list)
    for item_id, event in starts.items():
        grouped[(event['config'], event['entry_point'], event['variant'])].append(item_id)
    groups = []
    for (config, entry_point, variant), ids in sorted(grouped.items()):
        selected = set(ids)
        terminal = [ends[i] for i in ids if i in ends]
        completed = [e for e in terminal if e['status'] == 'ok']
        matched = [e for e in completed if e.get('category_kind') != 'unmatched']
        provider_events = [e for e in attempts if selected.intersection(e['item_ids'])]
        costs, embeddings, unknown = 0., 0., 0
        stage_times = defaultdict(list)
        costs_by_user = defaultdict(float)
        users = defaultdict(list)
        for i in ids:
            users[starts[i]['cohort']].append(i)
        for event in provider_events:
            stage_times[event['stage']].append(event['elapsed_ms'])
            cost = (event.get('usage') or {}).get('cost')
            if type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0:
                unknown += 1
                continue
            share = len(selected.intersection(event['item_ids'])) / len(event['item_ids'])
            if event['stage'] == 'embedding':
                embeddings += cost * share
            else:
                costs += cost * share
                for item_id in selected.intersection(event['item_ids']):
                    costs_by_user[starts[item_id]['cohort']] += cost / len(event['item_ids'])
        reviewed = [quality[i] for i in ids if i in quality and i in ends and ends[i]['status'] == 'ok']
        summaries_reviewed = [q for q in reviewed if q['summary_correct'] is not None]
        request_ids = {starts[i]['request_id'] for i in ids}
        fallback = [e for e in terminal if e.get('path') == 'control_fallback']
        sdk_hidden = sum(e.get('retry_visibility') == 'sdk_internal' for e in provider_events)
        group = dict(config=config, entry_point=entry_point, variant=variant,
            unique_users=len(users), attempted_items=len(ids), completed_items=len(completed),
            failed_items=sum(e['status'] != 'ok' for e in terminal), incomplete_items=len(ids)-len(terminal),
            classified_coverage=ratio(len(matched), len(completed)),
            fallback_rate=ratio(len(fallback), len(ids)),
            fallback_stages=dict(Counter(e.get('fallback_stage', 'unknown') for e in fallback)),
            native_treatment_completed=sum(e.get('path') == 'treatment' for e in completed),
            summary_attempts=sum(e['stage'] == 'summary' for e in provider_events),
            summary_email_rate=ratio(sum(e.get('summary_needed', 0) > 0 for e in terminal), len(ids)),
            item_latency=durations([e['elapsed_ms'] for e in terminal]),
            endpoint_latency=durations([requests[r]['elapsed_ms'] for r in request_ids if r in requests]),
            stage_latency={k: durations(v) for k, v in stage_times.items()},
            control_batch_sizes=dict(Counter(str(e.get('batch_size', 1)) for e in provider_events if e['stage'] == 'control')),
            known_inference_cost=costs, known_embedding_cost=embeddings,
            inference_cost_per_item=costs / len(ids) if not unknown and len(terminal) == len(ids) and provider_events else None,
            cost_coverage=ratio(len(provider_events)-unknown, len(provider_events)),
            unknown_cost_attempts=unknown, opaque_sdk_call_count=sdk_hidden,
            reviewed_items=len(reviewed), classification_accuracy=ratio(sum(q['category_correct'] for q in reviewed), len(reviewed)),
            reply_accuracy=ratio(sum(q['response_required_correct'] for q in reviewed), len(reviewed)),
            reviewed_summaries=len(summaries_reviewed),
            summary_accuracy=ratio(sum(q['summary_correct'] for q in summaries_reviewed), len(summaries_reviewed)),
            user_balanced_known_cost_per_item=mean(costs_by_user[u] / len(user_ids) for u, user_ids in users.items()),
            user_balanced_failure_rate=mean(sum(i in ends and ends[i]['status'] != 'ok' for i in user_ids)/len(user_ids)
                                           for user_ids in users.values()))
        groups.append(group)
    comparisons = []
    paired = defaultdict(dict)
    for group in groups:
        paired[(group['config'], group['entry_point'])][group['variant']] = group
    for (config, entry_point), variants in sorted(paired.items()):
        if not {'control', 'treatment'} <= variants.keys():
            continue
        control, treatment = variants['control'], variants['treatment']
        differences = {}
        for key in ('inference_cost_per_item', 'classification_accuracy', 'reply_accuracy',
                    'summary_accuracy', 'fallback_rate', 'classified_coverage'):
            a, b = control[key], treatment[key]
            differences[key] = b - a if a is not None and b is not None else None
        for key in ('p50_ms', 'p95_ms'):
            a, b = control['item_latency'][key], treatment['item_latency'][key]
            differences['item_latency_' + key] = b - a if a is not None and b is not None else None
        comparisons.append(dict(config=config, entry_point=entry_point, treatment_minus_control=differences))
    return dict(experiment=experiment, start=beginning.isoformat(), end=ending.isoformat(),
        groups=groups, comparisons=comparisons, invalid_lines=invalid, duplicate_events=duplicates,
        orphan_terminals=len(set(ends)-set(starts)), unmatched_quality_records=len(set(quality)-set(starts)),
        observed_unhealthy_events=sum(e.get('telemetry_healthy') is False for e in events),
        notes=[
            'Descriptive report; no automatic winner or statistical significance claim.',
            'Accuracy and summary quality require blinded reviewer records; confidence is not accuracy.',
            'Unknown costs and opaque SDK retries prevent exact invoice/savings claims.',
            'Inference excludes Pinecone, infrastructure, taxes and credit-purchase fees. Embeddings are separate.',
            'Batch cost is recorded once; user-balanced batch allocation divides cost equally among items.',
            'Mixed-batch endpoint latency is shared across assigned cohorts; item timings include queueing.',
            'Restart/disk gaps and requests crossing the window may produce incomplete/orphan observations.',
            'The existing correction endpoint cannot attribute corrections to a historical prediction.'])


def write_report(report, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    date = utc(report['end']).date().isoformat()
    md_path, json_path = root / f'ab-weekly-{date}.md', root / f'ab-weekly-{date}.json'
    def value(v):
        return 'not measured' if v is None else f'{v:.6g}' if isinstance(v, float) else str(v)
    lines = [f"# A/B report: {report['experiment']}", '',
             f"UTC window: {report['start']} to {report['end']} (end exclusive).", '',
             f"Invalid lines: {report['invalid_lines']}; duplicates: {report['duplicate_events']}; "
             f"unhealthy events: {report['observed_unhealthy_events']}.", '']
    if not report['groups']:
        lines += ['No data in this window.', '']
    for group in report['groups']:
        lines += [f"## {group['variant']} / {group['entry_point']} / {group['config']}", '',
                  '| Metric | Value |', '|---|---|']
        for key in ('unique_users', 'attempted_items', 'completed_items', 'failed_items', 'incomplete_items',
                    'classified_coverage', 'fallback_rate', 'summary_attempts', 'summary_email_rate',
                    'known_inference_cost', 'known_embedding_cost', 'inference_cost_per_item', 'cost_coverage',
                    'unknown_cost_attempts', 'opaque_sdk_call_count', 'reviewed_items', 'classification_accuracy',
                    'reply_accuracy', 'reviewed_summaries', 'summary_accuracy', 'user_balanced_known_cost_per_item'):
            lines.append(f"| {key.replace('_', ' ')} | {value(group[key])} |")
        for name in ('item_latency', 'endpoint_latency'):
            for quantile, number in group[name].items():
                lines.append(f"| {name} {quantile} | {value(number)} |")
        lines += ['', 'Stage latency (ms): ' + json.dumps(group['stage_latency']), '',
                  'Fallback stages: ' + json.dumps(group['fallback_stages']), '']
    for comparison in report['comparisons']:
        lines += [f"## Treatment minus control / {comparison['entry_point']} / {comparison['config']}", '',
                  '| Metric | Difference |', '|---|---|']
        lines += [f"| {key.replace('_', ' ')} | {value(number)} |"
                  for key, number in comparison['treatment_minus_control'].items()]
        lines += ['', 'Positive differences mean higher treatment values; accuracy rates are fractions, costs are USD.', '']
    lines += ['## Interpretation', ''] + ['- '+note for note in report['notes']]
    # Replace atomically so interrupted report generation cannot leave half a file.
    for path, content in [(md_path, '\n'.join(lines)+'\n'),
                          (json_path, json.dumps(report, indent=2, allow_nan=False)+'\n')]:
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(content, encoding='utf-8')
        temp.replace(path)
    return md_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=os.getenv('AB_DATA_DIR', 'data/ab'))
    parser.add_argument('--output-dir', default=os.getenv('AB_REPORT_DIR', 'reports'))
    parser.add_argument('--experiment-id', default=os.getenv('CLASSIFICATION_AB_EXPERIMENT_ID', 'clef-flash-v1'))
    parser.add_argument('--start', required=True, help='UTC start date/time, inclusive')
    parser.add_argument('--end', required=True, help='UTC end date/time, exclusive')
    parser.add_argument('--quality-file', help='Private JSON array of reviewer records')
    args = parser.parse_args()
    try:
        report = build_report(args.data_dir, args.experiment_id, args.start, args.end, args.quality_file)
        paths = write_report(report, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
