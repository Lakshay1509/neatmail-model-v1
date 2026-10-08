# Spec: Clef Flash classification A/B experiment

Status: implemented, offline verified; disabled pending live calibration and deployment checks.
Documentation checked: 2026-10-08.

## Objective

Implement working traffic routing in the existing FastAPI application to compare
the current GPT-5 nano classifier with Clef Flash classification followed by GPT-5 nano
only when summary/action generation is needed. Preserve the public response
contracts and per-user correction retrieval.

## Confirmed user choices

- Working code wired into the API, not documentation alone.
- 50% control / 50% treatment, stable per user.
- Control continues calling OpenAI directly.
- Treatment calls Clef Flash and GPT-5 nano through OpenRouter.
- Deployment is one server/container on Coolify with persistent disk.
- Weekly reports belong in `reports/` as Markdown and JSON.

Assignment balances users approximately, not exactly half of email volume: users
send different numbers of emails. Since provider and architecture both change,
this experiment measures the whole new path, not Clef Flash alone.

## Tech stack

Existing dependencies: FastAPI 0.135.2, Pydantic 2.12.5, OpenAI SDK 2.30.0,
httpx 0.28.1, Pinecone 8.1.0. No new dependency is proposed. Use existing httpx
for both Decisions requests and OpenRouter Chat Completions summaries.

## Behavior

### Assignment and configuration

- New `classification_ab.py` owns deterministic assignment and treatment calls;
  it does not import `main.py`, avoiding circular imports and startup side effects.
- Hash experiment ID plus `user_id` with SHA-256; map the hash to a bucket.
  Serialize the pair unambiguously (e.g. a JSON array), rather than concatenating
  strings that can collide. Do not use Python's randomized built-in `hash()`.
- The same user receives the same variant across requests, restarts, workers,
  and `/classify` versus `/classify-batch`, while configuration is unchanged.
- Configuration: `CLASSIFICATION_AB_ENABLED` (default false),
  `CLASSIFICATION_AB_TREATMENT_PERCENT` (default 50),
  `CLASSIFICATION_AB_EXPERIMENT_ID` (default `jev-v2`),
  `OPENROUTER_API_KEY`, `OPENROUTER_DECISION_MODEL` (default `typesafe/jev-1.13`),
  `OPENROUTER_DECISION_PROVIDER` (default `auto`),
  and `OPENROUTER_SUMMARY_MODEL` (default `openai/gpt-5-nano`).
- Additional settings: `AB_DATA_DIR` (local default `data/ab`),
  `AB_REPORT_DIR` (local default `reports`), `AB_COHORT_HMAC_KEY` (secret for
  pseudonymous telemetry IDs), `AB_CATEGORY_MIN_PROBABILITY` (0.4, calibrated 2026-10-08),
  and `AB_NOUL_THRESHOLD` (initially 0.5).
- Validate configuration explicitly. Enabled nonzero treatment requires an OpenRouter
  key; invalid settings must not silently turn the experiment into control.
- Enabling telemetry requires a stable HMAC key and writable data directory.
- Disabling the experiment routes all requests to the original implementation
  and does not require an OpenRouter key. Setting treatment percentage to zero
  provides another rollback option and also does not require that key.
- Preserve experiment ID, split, HMAC key, models, prompts, and thresholds during
  the measurement window. Record a configuration fingerprint and code revision;
  changed configurations appear in separate report segments, never silently pooled.
- Keys remain server-side. Do not read, overwrite, or commit actual `.env` values.

### Control

Preserve the current OpenAI GPT-5 nano classification prompt, model parameters,
structured output, normalization, summary gating, and digest suppression.
Embeddings and Pinecone remain as currently implemented for both variants.

### Treatment

- Retrieve corrections once per email, as for control. Send only necessary email
  fields, sensitivity, tag definitions, and that user's relevant corrections.
  Do not send the user ID or experiment assignment to model providers.
- Call `POST https://openrouter.ai/api/alpha/decisions` with Bearer authentication,
  the pinned Clef Flash model, `state`, and named typed `questions`.
- Decompose policy into focused questions: correction applicability, automated
  sender, cold outreach, actual reply/action obligation, topic selection, and
  human judgment needed for summary generation. Use Choice and Noul answers.
- Questions are evaluated independently. A question cannot read another answer
  from the same call. Ask speculative questions against the same state and combine
  only in Python. Select a matching correction by its opaque option ID, including
  a no-match option, then resolve its label locally against supplied tags.
- Apply policy in Python: applicable corrections take precedence; cold outreach
  cannot become Pending Response or Action Needed solely because of a sales ask;
  genuine human obligations and automated decisions outrank topic selection.
- Respect dynamically supplied tags and preference for user-defined topic tags.
  Use an explicit unmatched option and map it back to `category=""`.
- Use internal option IDs to avoid collision between a user tag and the unmatched
  sentinel. Omit correction questions when there are no usable corrections. Empty
  tags yield an unmatched category; never force a Choice with invalid option count.
  Retain an application safety cap of 255 Choice options, including sentinels;
  oversized/unsupported requests take the recorded fallback path, not truncation.
- Validate answer types, required fields, allowed choices, finite probabilities,
  and probability ranges. Do not convert malformed responses to plausible defaults.
- Preserve sensitivity behavior, automated/cold-outreach reply suppression,
  and digest summary/action suppression.
- Category threshold: selected option probability >= 0.4;
  otherwise return an empty-category result. Noul binary decisions use
  a configurable threshold initially 0.5. These are provisional policy settings,
  not claims of measured accuracy. Log probability and optional confidence separately; absence of confidence is valid.
- The current control's 95% condition is a prompt instruction, not a numeric
  measurement. The treatment cutoff is not established as equivalent. Test
  thresholds on a calibration set before the live experiment and freeze them;
  keep held-out quality evaluation separate. Report accuracy AND classified
  coverage so abstaining frequently cannot manufacture a good accuracy score.
  A topic Choice probability is not the confidence of the composed final result;
  report each contributing answer separately and do not multiply independent
  question probabilities to invent a final calibrated probability.
- Generate `ai_summary` and `ai_action` via OpenRouter GPT-5 nano only for final
  Pending Response, or Action Needed requiring human judgment. Never call the
  summary model for digest emails, unmatched categories, ordinary topic tags,
  or one-click triggers that should have empty AI fields.
- The summary call returns only summary/action fields under a strict JSON schema;
  it cannot overwrite category or response_required. Reuse the current summary
  wording and allowed action list.
- Use `https://openrouter.ai/api/v1` as the summary client's base URL. For the
  Chat Completions request require provider support for parameters with
  `provider.require_parameters=true` in the plain HTTP JSON request. Use
  OpenRouter's documented `reasoning` object and `max_tokens` for the treatment
  request rather than copying all direct-OpenAI parameters blindly. Start with
  medium effort and a 6000-token reasoning-plus-output budget; inspect refusals,
  empty content, and finish reason. Validate summary strings and action enum
  locally; do not truncate text to enforce word counts.
- Use bounded timeouts and retries for retryable provider errors. Validate
  structured responses locally even when the provider claims schema support.
- OpenRouter usage is automatic; do not add the deprecated
  `usage: {include: true}` flag. Keep the two response shapes distinct: Decisions
  uses `input_tokens`/`output_tokens`; Chat Completions uses
  `prompt_tokens`/`completion_tokens` and token detail fields.
- Configure connect/read/write/pool timeouts explicitly. HTTPX's read timeout
  measures inactivity, not total wall time. Set an explicit experimental attempt
  budget (initially two attempts per stage), capped backoff honoring Retry-After,
  and a monotonic total treatment budget (initially 60 seconds before fallback).
  Pass remaining budget into each attempt and refuse new calls/backoff once it
  is exhausted; waiting on a thread future is not cancellation of its HTTP call.
  Never launch fallback while a timed-out experimental thread is still issuing
  background model requests.
  Do not claim a hard whole-request SLA: the original OpenAI fallback retains its
  own timeout behavior. Both treatment APIs use plain httpx with one retry owner,
  so nested SDK retries cannot multiply calls. Retry
  only transient failures (network, 408, 429, retryable 5xx), not bad credentials,
  insufficient credits, invalid schema, or unsupported parameters. Record attempts.

### Failure handling

On a treatment provider error, timeout, or invalid response, fall back to the
existing OpenAI classifier for that email. Avoid recursive routing. Log assigned
variant `treatment`, executed path `control_fallback`, failure stage, and all
known incurred cost/latency. If fallback fails, return the existing error behavior.
Prepared corrections must be passed into the control fallback rather than embedded
and queried again. Capture usage immediately after every model response, before
validating its content, so malformed/empty completions still count as expenditure.
Summary-stage failure follows the same recorded full-control fallback policy;
the report distinguishes it from Clef Flash failure and accounts for both stages.
Low-confidence classifications return empty category rather than automatically
calling another classifier; this keeps summary-only nano routing in scope.

### Batch requests

- Partition requests by each item's user assignment, including mixed-user batches.
- Continue using the existing batch classifier for control items together.
- Process treatment items individually with bounded concurrency and isolated
  corrections/tags. Do not silently serialize all treatment emails if a batch
  can be handled concurrently.
- Restore input order and IDs after merging; preserve empty-batch and maximum
  ten-email validation. Never drop or duplicate results because of partitioning.
- Merge by original input position, not only an ID dictionary (incoming IDs may
  repeat). Give the control sub-batch unique internal IDs and map back to originals.
  Missing/duplicate/unknown provider IDs are validation failures, never additional
  output rows or fabricated successes. Keep the experiment-disabled legacy path.
- Evaluate treatment failures per email; do not rerun successful treatment items.
- Use a single bounded executor per process (initial maximum four workers), not
  an unbounded pool created per request. Pool limits apply across concurrent
  batches. Instrument partial failure even if the public batch request fails.
- Splitting a batch reduces control batch size and can change cost and quality.
  Report single and batch endpoints separately, including control sub-batch size;
  do not interpret that effect as intrinsic Clef Flash model performance.

## Measurement

The telemetry must answer: is treatment cheaper, faster, reliable, and accurate
enough at useful coverage? Emit structured application logs with experiment ID,
assigned variant, executed path, opaque user cohort key, request correlation ID,
entry point, item count, fixed category kind (not raw custom tag text),
response_required, confidence/probability, summary-call count, elapsed time,
provider/model, usage tokens, known cost, and fallback/error stage.
Never log email subject/body, sender address, corrections, API keys, or raw
provider error text that may contain request content. Unknown cost stays unknown,
not zero. Custom tags can contain personal data; log a fixed kind such as
pending_response/action_needed/topic/unmatched rather than their names.
Use HMAC-SHA256 for telemetry cohort IDs (separate from assignment hashing), not
plain hashes of predictable user IDs. Keep pseudonymous records private.
Record per-stage latency, per-item processing latency, and endpoint elapsed time
including queueing, embeddings, Pinecone, all model attempts, and fallback.

### Durable event files and cost accounting

- Write schema-versioned JSONL events under `AB_DATA_DIR`, rotated daily, with a
  separate filename per process boot UUID. Serialize thread writes with a lock;
  never have several processes append to a single file without coordination.
  Flush records promptly. Record UTC timestamps, unique event IDs, request IDs,
  generated item trace IDs, stage/attempt IDs, and configuration fingerprint.
- Emit request/item start and terminal events, plus provider-attempt events.
  Missing terminal events after crashes become incomplete observations. Deduplicate
  event IDs when loading exports and report malformed/truncated-line counts.
- Treat provider-attempt events as the cost ledger. Terminal events link to
  attempts and do not repeat billable totals. One shared control batch completion
  is charged once, not once per item. Equal division among its items is an explicit
  allocation estimate for per-user views; retain the exact batch total.
- Prefer OpenRouter `usage.cost` as reported inference cost. For direct OpenAI,
  compute an estimate using recorded pricing version plus uncached input, cached
  input, and total completion tokens. Reasoning tokens are already inside the
  completion count; never add them twice. Missing usage or a lost response leaves
  cost incomplete. Include failed/retried attempts with known usage and mark
  unknown costs, rather than assuming failures are free.
- Include known embedding costs separately. Pinecone/infrastructure costs are
  outside per-call inference billing and must be marked excluded. Label the cost
  comparison as reported/estimated inference cost, not the final account invoice;
  provider credit-purchase fees and taxes are excluded. Show cost coverage and
  known-cost subtotals; with missing usage do not claim exact percentage savings.
- Check directory permissions at startup. Write failures must emit a sanitized
  stderr warning and mark telemetry unhealthy without recursively writing to the
  failing sink. Reports expose start/terminal mismatches and observed gaps; a
  total disk outage can lose events entirely and requires operator verification.

Compare p50/p95 total latency, model cost per email, summary-call rate,
empty-category rate, error/fallback rate, and labeled classification accuracy.
Attribute fallbacks to their original treatment assignment in primary analysis.
Analyze results with users as the assignment unit. Live A/B routing does not
produce paired predictions; use labeled evaluation cases for direct accuracy
comparison. User corrections alone are incomplete and delayed quality feedback.

### Weekly report

- Add `eval/report_ab.py` to read events offline, filter by experiment and UTC
  start-inclusive/end-exclusive dates, and write both
  `reports/ab-weekly-YYYY-MM-DD.md` and `.json`. Date is the window's end date.
  Use configured report directory; Coolify destination is `/app/reports`.
- Generation is a manual command in this initial scope, not an automatic scheduler
  or a new web endpoint. Reports are private server files; download them from the
  server/volume. Local runs save in the local project's `reports/` directory.
- Show unique users, attempted/completed/failed/incomplete items, summary-call
  rates, fallback by stage, unmatched rate, per-stage and endpoint p50/p95 latency,
  cost coverage, known-cost totals/per-item costs, and configuration changes.
  No observations yields "no data", not a zero-error or zero-cost result.
- Group primary results by assigned variant including fallbacks. Show native
  treatment results separately as diagnostics. Give volume-weighted business
  metrics and user-balanced views; separate single requests from batches.
- Report descriptive differences and sample sizes. Do not declare a winner from
  a week or from model confidence. Statistical uncertainty must respect user
  assignment (e.g. resample users rather than individual emails). With few users,
  missing usage, or incomplete quality data, state that evidence is insufficient.
- Optional `--quality-file` accepts private, content-free reviewer records keyed
  by generated item trace ID: category_correct, response_required_correct,
  summary_correct (nullable), and reviewer provenance. Check joins/deduplication
  and include reviewed-sample counts. Without reviewer data, accuracy/summary
  quality are "not measured"; telemetry alone cannot judge semantic correctness.
  Human review occurs in the existing email system with variant hidden, using a
  random sample from both cohorts; do not copy email bodies into experiment logs.
- `/correct` currently has no prediction ID or variant linkage. It cannot produce
  a valid per-variant correction rate without an additional interface. Do not
  change that API in this scope or infer attribution from current user assignment.

### Coolify deployment instructions

Use the existing Dockerfile build and container port 8000. In the application's
Configuration > Persistent Storage add two Volume Mounts with Destination Paths
`/app/data/ab` and `/app/reports`. In runtime environment variables set
`AB_DATA_DIR=/app/data/ab` and `AB_REPORT_DIR=/app/reports`, the API/HMAC keys,
experiment ID, enabled flag, and 50% allocation. Keys are runtime secrets, not
Docker build arguments. Do not mount over `/app` itself, which would hide code.

Attach storage before enabling the experiment. Verify a write is readable after
a redeployment and the running application user can write both paths. Keep this
initial experiment on one server; local volumes do not become shared across
servers. Reports and telemetry are not exposed by an HTTP static-file route.
Include backup/retention instructions without deleting data automatically; named
volumes survive container replacement but are not backups against server loss.

## Implementation status

Implemented with the experiment disabled by default. Both treatment stages use
plain httpx; routing lives in `ab_routing.py`. The existing OpenAI SDK retries
remain unchanged and are marked opaque in the ledger: known costs are a subtotal,
not a guaranteed invoice total. Returned provider identifiers are bounded and
validated before logging.

Enabled API responses include `X-AB-Item-IDs` (one opaque ID per email, ordered
for batches) for joining blinded reviewer labels. Response JSON remains unchanged.
Reports contain descriptive treatment-minus-control differences for matching
configuration and endpoint segments; incomplete observations cannot establish a
complete cost per item.

Verified with 32 offline tests, compilation, diff checks, a regression mutation
check, and the actual report CLI using synthetic data. Live model calibration,
account/model access and Coolify volume persistence remain operator checks before
enabling. No deployment or paid model call was performed.
The final production pass adds a Docker image health check, sanitized control
errors, bounded retry for documented transient credit holds, patched dependency
pins and GitHub CI. See `tasks/production-readiness.md` for evidence and remaining
deployment checks.

## Project structure

- `classification_ab.py`: assignment, treatment provider calls, validation.
- `ab_routing.py`: bounded single/batch orchestration, fallback and attribution.
- `main.py`: narrow integration in single and batch classification paths.
- `tests/test_classification_ab.py`: offline routing and integration tests.
- `ab_metrics.py`: bounded, private JSONL event writer and usage normalization.
- `eval/report_ab.py`: offline weekly report command.
- `tests/test_ab_report.py`: aggregation, dates, costs, and quality-join tests.
- `.env.example`: example experiment settings, without secrets.
- `eval/README.md`: operation, rollback, Coolify mounts, report command, evaluation
  instructions and limitations. Ignore generated `data/ab/` and `reports/` in git.
- `SPEC-classification-ab.md`: this reviewed behavior contract and source links.

## Code style

Match existing snake_case Python and Pydantic contracts. Prefer small functions
and explicit parameters over an experiment framework or provider registry.

```python
def assign_variant(user_id: str, experiment_id: str, treatment_percent: int) -> str:
    # Stable across processes; assignment never uses email content.
    ...
```

## Commands and verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py classification_ab.py ab_routing.py ab_metrics.py eval/report_ab.py
uvicorn main:app --host 127.0.0.1 --port 8000
python eval/run_eval.py --threshold 1.0 --json
python eval/report_ab.py --data-dir data/ab --output-dir reports --experiment-id jev-v2 --start 2026-10-08 --end 2026-10-15
```

Coolify container terminal example (dates must match the actual test window):

```sh
python eval/report_ab.py --data-dir /app/data/ab --output-dir /app/reports --experiment-id jev-v2 --start 2026-10-08 --end 2026-10-15
```

Write offline tests first for stable assignment, 0/100 endpoints, invalid config,
experiment disabled, both provider routes, summary gating, correction precedence,
digest suppression, invalid responses, provider failures/fallback attribution,
and mixed-user batches. Mock Pinecone and model clients: unit tests make no paid
calls and do not import real startup clients. Use the existing labeled harness
with separate all-control and all-treatment server configurations for live model
evaluation; preserve its existing threshold. Document when credentials or a
running service prevent live verification rather than claiming it passed.
The existing harness exercises `/classify` only, despite its README implying
shared-prompt coverage is enough for batches. Add explicit mixed/same-user batch
tests and summary-gating cases. Test multiple processes/restarts, corrected
configuration rollback, internal ID collisions, unavailable storage, empty report
windows, UTC boundaries, duplicated/truncated events, missing costs, and single
counting of batch/failed-attempt charges. Generated report tests use synthetic
events and verify both file formats, with no secrets or email text.

Before live allocation, smoke-test actual account access to both requested models
and schema support; documentation cannot establish account availability. GPT-5
nano's official model page is marked deprecated, but retain the user's requested
model and current control rather than silently replacing it. Log returned model
versions; configuration or model changes invalidate pooled comparisons.

## Boundaries

- Always: preserve API/auth contracts, isolate user corrections, verify offline
  routing, document configuration and sources, report verification limits.
- Ask first: changes to public response fields, new persistent storage, additional
  dependencies, or changes to the agreed allocation/provider choices.
- Never: log secrets/email content, replace existing embeddings, alter unrelated
  prompts or endpoints, lower evaluation thresholds, deploy without instruction.

## Success criteria

1. Both endpoints actually use the configured experiment with stable assignment.
2. Control remains on OpenAI; treatment uses OpenRouter for both model stages.
3. Non-summary treatment emails incur no GPT-5 nano summary call.
4. Existing response fields and per-user corrections remain compatible.
5. Offline tests cover routing, summary gating, failure paths, and batch merging.
6. Logs distinguish original assignment, execution, fallback, and measured usage.
7. Operators can turn the experiment off without requiring OpenRouter credentials.
8. JSONL measurements and generated Markdown/JSON reports survive Coolify redeploys.
9. One offline command generates a dated report with honest cost/quality coverage,
   no duplicate batch billing, and visible fallback/incomplete observations.

## Review decisions

Allocation, stability, provider selection, implementation scope, and one-server
Coolify deployment are confirmed. Fallback-on-error remains the recorded design;
thresholds require calibration, not an assertion of 95% real-world accuracy.

## Review findings corrected (2026-10-08)

1. Added durable measurements and the promised report files/command.
2. Defined Coolify volume destinations and redeployment verification.
3. Prevented double-counting batch and reasoning-token costs; retained unknowns.
4. Added OpenRouter parameter support, automatic usage, and explicit retry ownership.
5. Separated calibration, confidence, classified coverage, and reviewed accuracy.
6. Fixed zero-percent rollback credentials, independent-question assumptions,
   option/ID collisions, fallback correction reuse, and global concurrency limits.
7. Removed raw custom-category names from telemetry; used keyed cohort IDs.
8. Identified existing eval coverage limits and absent correction attribution.

The initial findings were a design review; runtime routing is now implemented and
verified with mocked providers. No paid provider calls or Coolify deployment have
been performed.

## Clef migration decision (2026-10-08)

User selected Clef Flash instead of Jev and explicitly chose Prime Intellect.
Decisions requests send `provider={"only":["primeintellect"],"allow_fallbacks":false}`.
The endpoint advertises 16,384 context tokens and $0.021/M input, zero output
pricing. Workers AI documents roughly 2,000 text-state-token truncation despite
its larger advertised context. Provider failure uses the existing recorded
OpenAI control fallback; it does not silently switch Clef providers.

Remove obsolete `OPENROUTER_JEV_MODEL`; startup rejects it. Use a fresh
`clef-flash-v1` experiment ID to separate historical Jev measurements. This can
reassign users when migrating; assignment remains stable within the experiment.
Provider changes also change the configuration fingerprint. Recalibrate on
representative labeled emails before enabling: previous Jev results and mocked
routing tests cannot prove Clef accuracy. The old Linux image verification
predates this migration; current checks are recorded separately in task evidence.

## Jev re-switch decision (2026-10-08)

Benchmarked on 50 graded cases (`eval/clef_cases.json`), decisions only, one run:
Jev 1.13 48/50, Clef Flash 42/50, gpt-5-nano control prompt 43/50; Jev p50 0.7s
and ~9% of control's classification cost. Defaults switched to `typesafe/jev-1.13`,
provider `auto`, experiment `jev-v2`, category threshold 0.4 (0.95 blanked correct
topics; no wrong topic was blocked by any cutoff up to 0.40). Clef remains
configurable via `OPENROUTER_DECISION_MODEL` with `OPENROUTER_DECISION_PROVIDER=primeintellect`.

## Official sources

- Clef Flash and Cloudflare truncation notice:
  https://openrouter.ai/cloudflare/clef-flash
- Clef provider contexts and prices:
  https://openrouter.ai/api/v1/models/cloudflare/clef-flash/endpoints

- Decisions request/response endpoint and model ID:
  https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request
- Structured output and provider parameter support:
  https://openrouter.ai/docs/guides/features/structured-outputs
- OpenRouter GPT-5 nano model:
  https://openrouter.ai/openai/gpt-5-nano
- OpenAI-compatible OpenRouter base URL:
  https://openrouter.ai/docs/guides/community/openai-sdk
- Usage/cost accounting and deprecated include flag:
  https://openrouter.ai/docs/cookbook/administration/usage-accounting
- OpenRouter reasoning parameter and output billing:
  https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
- Direct OpenAI cached-input pricing and requested-model status:
  https://developers.openai.com/api/docs/models/gpt-5-nano
- HTTPX timeout semantics:
  https://www.python-httpx.org/advanced/timeouts/
- SDK default retries (implementation must verify against pinned 2.30.0):
  https://github.com/openai/openai-python#retries
- Coolify application mount configuration:
  https://coolify.io/docs/applications/configuration/persistent-storage
- Coolify volume mount verification:
  https://coolify.io/docs/core/persistent-storage/storage-mounts/volume-mounts
- Coolify persistence boundaries:
  https://coolify.io/docs/core/persistent-storage/storage-mounts/overview
