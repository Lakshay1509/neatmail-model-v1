# Classifier eval harness

Regression check for the email classifier. Sends labeled emails to the running
`/classify` endpoint and scores the predicted `category` / `response_required`
against expectations.

It scores **both directions**:
- `Pending Response` fires when a human directly asks the recipient to reply/act
  (8 cases, incl. the 4 misclassified screenshots).
- `Pending Response` does **not** over-fire on automated / informational mail that
  merely mentions payments, files, or receipts (10 counter-cases) — this is the
  guard against the fix swinging too far the other way.
- `Pending Response` does **not** fire on **cold outreach** (6 cases): sales pitches,
  recruiter mail, partnership proposals and sequence follow-ups that are hand-written
  and end in a direct question, but that the recipient owes nothing. Two of the
  8 fire-cases (`inbound-prospect-pricing`, `reply-to-my-demo-request`) are the guard
  in the other direction — a stranger the recipient *does* owe a reply.

## Run

```bash
# 1. start the API (needs OPENAI_API_KEY, PINECONE_API_KEY, DASHBOARD_API_KEY):
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. in another terminal:
export DASHBOARD_API_KEY=...        # same key the server expects
python eval/run_eval.py             # full run, must be 100% to exit 0
python eval/run_eval.py --only swiggy-receipt-ask,forward-latest-file
python eval/run_eval.py --threshold 0.9 --json
```

Exit code is `0` if the pass rate ≥ `--threshold` (default `1.0`), else `1`, so it
can gate CI.

## Notes

- Uses `user_id=eval-harness` (override with `NEATMAIL_EVAL_USER`). Use a user with
  **no stored corrections** so the eval measures the prompt, not correction memory.
- This harness exercises `/classify` only. Offline API tests separately verify
  `/classify-batch` routing, ordering and validation; shared prompts alone do not
  establish batch correctness.
- The model is stochastic-ish even at `seed=42`; a case that flips between runs is a
  low-confidence signal worth a prompt tweak, not a flaky test to ignore.

## Adding cases

Edit `cases.json`. A case inherits `default_tags` unless it sets its own `tags`.
Assertions under `expect`:

| key                 | meaning                                                        |
|---------------------|----------------------------------------------------------------|
| `category`          | exact match; a **list** means any-of is acceptable             |
| `not_category`      | prediction must **not** equal this; a **list** means none-of   |
| `response_required` | optional boolean check                                          |

## Jev A/B experiment

The experiment is disabled by default. When enabled, stable SHA-256 assignment
routes approximately 50% of users to each variant; email volume need not be 50/50.
The control uses the existing OpenAI `gpt-5-nano` prompt. Treatment uses
OpenRouter `typesafe/jev-1.13` for decisions and `openai/gpt-5-nano` only for
needed summary/action text. Failures in Jev or summary generation fall back to
the original OpenAI classifier; metrics retain treatment assignment and failure
stage. No user-facing JSON fields have changed.

Settings are shown in `.env.example`. Copy it locally if needed and set real
keys outside git. Generate the HMAC secret once with:

```sh
python -c "import secrets; print(secrets.token_hex(32))"
```

Keep the HMAC key, experiment ID, percentage, model names and thresholds stable
for the week. Changing policy/configuration creates a separate report segment.
Model confidence is not measured accuracy. The 0.40 category-probability and
0.5 binary thresholds are policies; recalibrate when the model changes. Decisions
also use binary answers, so the category threshold is not a guarantee for the
entire pipeline. Choice confidence is optional; the cutoff uses selected-option
probabilities.

**Benchmark (2026-10-08)** on `eval/clef_cases.json` (50 cases, 5 difficulty levels),
one run each, no corrections, decisions only (summary call excluded from cost):

| | Jev 1.13 | Clef Flash | gpt-5-nano (control prompt) |
|---|---|---|---|
| Correct | **48/50** | 42/50 | 43/50 |
| p50 / p95 latency | 0.7s / 2.1s | 0.9s / 2.1s | 14.3s / 28.2s |
| Cost per 50 emails | $0.0024 | $0.0013 | $0.027 |

The 0.40 cutoff: raising it never blocked a wrong topic in `cases.json` or
`clef_cases.json`, only blanked correct ones; nothing is lost at 0.40 or below (at
0.95 Clef scored 25/50). Lowest observed selected-topic probability was 0.32;
chance with 6-8 tags is ~0.15. The same threshold also gates stored-correction
overrides, which this benchmark did not exercise.

`OPENROUTER_DECISION_PROVIDER=auto` lets OpenRouter route Jev. To run Clef Flash
instead, set `OPENROUTER_DECISION_MODEL=cloudflare/clef-flash` and pin
`OPENROUTER_DECISION_PROVIDER=primeintellect` (Cloudflare Workers AI documents
truncation to roughly the first 2,000 text-state tokens), with its own experiment ID.
Each treatment email has its own Decisions request, including corrections and
questions; batches are processed with bounded concurrency.

`OPENROUTER_JEV_MODEL` is obsolete; startup rejects it. Use
`OPENROUTER_DECISION_MODEL`. Experiment `jev-v2` is fresh so `jev-v1` and
`clef-flash-v1` observations are not pooled. Changing the ID can reassign users;
assignment stays stable within the new experiment.

### Coolify

Connect the Git repository containing these changes. Coolify builds committed
source from the selected branch; local uncommitted files are not included.
Select the Dockerfile build pack, Base Directory `/`, Dockerfile Location
`/Dockerfile`, and Ports Exposes `8000`. Keep the Dockerfile's existing CMD;
no separate install/build/start command is required. Use one server and one
Uvicorn process for this experiment. Add an HTTPS domain pointing to the server,
with its internal target port set to `8000`; public clients use HTTPS port 443.

Use the existing Dockerfile build, port **8000**. In **Configuration > Persistent
Storage**, add Volume Mounts at these exact Destination Paths:

| Data | Destination Path |
|---|---|
| Measurements | `/app/data/ab` |
| Generated reports | `/app/reports` |

Set these **runtime** environment variables in Coolify:

```dotenv
CLASSIFICATION_AB_ENABLED=true
CLASSIFICATION_AB_TREATMENT_PERCENT=50
CLASSIFICATION_AB_EXPERIMENT_ID=jev-v2
OPENROUTER_DECISION_MODEL=typesafe/jev-1.13
OPENROUTER_DECISION_PROVIDER=auto
AB_CATEGORY_MIN_PROBABILITY=0.4
AB_DATA_DIR=/app/data/ab
AB_REPORT_DIR=/app/reports
```

Also set `OPENROUTER_API_KEY`, `AB_COHORT_HMAC_KEY` and the existing OpenAI,
Pinecone and dashboard keys. Do not put secrets in build arguments. Attach the
mounts before enabling the experiment; a mount at `/app` would hide the code.
For each secret, select Runtime / Available in the container and disable Build
time availability. Preserve existing values when editing Coolify's Developer
view: saving it removes variables omitted from its contents. Start with
`CLASSIFICATION_AB_ENABLED=false`, verify the deployment and mounts, then enable
the experiment and redeploy/restart to apply the runtime environment.
In staging, write a test file into each destination, redeploy, and confirm both
remain readable/writable. Back up the volumes; persistence is not a backup.
Keep this setup on one server. Review disk growth and retain the full measurement
window plus time for review; the application does not delete events automatically.

The Dockerfile now owns the health check below. Coolify detects its `HEALTHCHECK`
and uses it in preference to a dashboard-configured check; no additional
dashboard check is needed. The app has no `/health` route and the image does not
need curl or wget:

```sh
python -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=4); assert r.status == 200"
```

Image timings: interval 30s, timeout 5s, retries 3, start period 120s (increase
in the Dockerfile if initial Pinecone index creation takes longer). This checks that FastAPI serves requests after startup;
it does not call paid models or prove provider readiness. `/openapi.json` is
FastAPI's built-in schema route and is accessible without `X-API-Key`; business
endpoints require that header. Test the exact command in the container terminal.

Rollback: set `CLASSIFICATION_AB_ENABLED=false` and redeploy. This restores the
legacy path and stops experiment measurements. Alternatively set percentage to
`0` to retain control measurements; no OpenRouter key is required in either case.
For the second option the HMAC key and measurement storage remain required.

### Run a weekly report

From the Coolify container terminal, adjusting dates to the actual week:

```sh
python eval/report_ab.py --data-dir /app/data/ab --output-dir /app/reports --experiment-id jev-v2 --start 2026-10-08 --end 2026-10-15
```

This writes private server files:

- `/app/reports/ab-weekly-2026-10-15.md`
- `/app/reports/ab-weekly-2026-10-15.json`

Download them from the server volume. They are not public API endpoints. Locally,
omit the directory flags to use `data/ab` and `reports`. Windows: activate `.venv`
or replace `python` with `.venv/Scripts/python`. The report is manual; no scheduler
has been installed. Repeating the same end-date report replaces those files.
Dates use UTC, start inclusive and end exclusive.

Compare unique users, inference/embedding costs, latency percentiles, classified
coverage, errors and fallback stages. Reports separate single calls, batches,
variants and configuration fingerprints. A batch's control completion is billed
once; per-user allocation splits that shared amount evenly. Missing usage is
unknown, not zero. Direct OpenAI costs are estimates with cached-input discounts;
reasoning tokens are already included in completion tokens. Legacy SDK internal
retries remain enabled and are not individually visible, so reported costs are
known/estimated subtotals, not an exact invoice. Pinecone, infrastructure, taxes
and credit-purchase fees are excluded. Failed/lost requests can have unobserved
charges. No savings or quality verdict is automatic.

### Quality reviews

With experiment enabled, `X-AB-Item-IDs` contains an opaque ID for `/classify`, or
one ID per input item in batch order. Have your calling application associate
these IDs privately with its emails if you want review attribution. No variant
is included in this header, allowing reviewers to remain blinded. Randomly
sample both cohorts for review in your existing email interface; do not copy
email content into telemetry or reports.

Pass a private JSON array to `--quality-file` to measure correctness:

```json
[
  {
    "item_id": "opaque-id-from-response-header",
    "category_correct": true,
    "response_required_correct": true,
    "summary_correct": null,
    "reviewer": "reviewer-1"
  }
]
```

Use `summary_correct: null` when no summary was reviewed. Duplicate IDs are
rejected. The report includes reviewed sample sizes and unmatched review counts;
without reviews, accuracy and summary quality are **not measured**. The current
`/correct` payload has no prediction ID and cannot supply reliable per-variant
correction rates. A week with few users or little reviewed data is inconclusive.

### Verification

GitHub Actions runs offline tests, compilation, the pinned dependency audit,
and a Docker build on pushes and pull requests. It does not deploy or use provider
secrets. Require the `verify` check before merging/deploying. Coolify's automatic
Git deployment does not by itself guarantee that CI has finished successfully;
deploy the verified commit manually or configure your own CI-success gate.

For local Podman verification, use Docker image format so Podman retains the
Dockerfile's `HEALTHCHECK` (its default OCI format drops it):

```sh
podman build --format docker --tag localhost/neatmail:production-check .
podman run --rm --network none --entrypoint python localhost/neatmail:production-check -m unittest discover -s tests -v
podman run --rm --network none --entrypoint python localhost/neatmail:production-check -m pip check
```

On Windows, a running Podman WSL machine is required. If the Windows remote
connection drops during a build, the same command can run directly inside that
machine's WSL distribution. The previous Jev production pass verified the Linux
build, image health check, mocked API, report CLI and named-volume persistence;
that evidence predates the Clef migration. The updated Clef Linux image also
builds and passes all 32 offline tests and `pip check`; its health-check
configuration is retained. On this rootless WSL machine, test containers required
`--cgroups=disabled` because the pids controller was unavailable. Live Clef access,
current Coolify storage and actual server health remain deployment checks.

```sh
python -m unittest discover -s tests -v
python -m compileall -q main.py classification_ab.py ab_routing.py ab_metrics.py eval/report_ab.py
```

These tests mock Pinecone/OpenAI startup and OpenRouter HTTP. They make no paid
calls. Before enabling production, run the labeled harness twice on staging:
once with treatment percentage `0`, then `100`, using the same cases and a user
without stored corrections. Keep `--threshold 1.0`. Review summary quality
separately. This also verifies actual account/model access and schema support,
which offline tests cannot establish. Direct OpenAI's model page currently marks
GPT-5 nano deprecated; the requested model is retained rather than replaced.

Official integration references:

- [Jev 1.13 model](https://openrouter.ai/typesafe/jev-1.13)
- [Clef Flash model and provider truncation notice](https://openrouter.ai/cloudflare/clef-flash)
- [Live Clef provider limits and prices](https://openrouter.ai/api/v1/models/cloudflare/clef-flash/endpoints)
- [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request)
- [OpenRouter structured output support](https://openrouter.ai/docs/guides/features/structured-outputs)
- [Automatic usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting)
- [Coolify persistent storage](https://coolify.io/docs/applications/configuration/persistent-storage)
- [Coolify Dockerfile deployment](https://coolify.io/docs/applications/build-packs/dockerfile)
- [Coolify runtime variable scope](https://coolify.io/docs/applications/configuration/environment-variables)
- [Coolify volume mount verification](https://coolify.io/docs/core/persistent-storage/storage-mounts/volume-mounts)
- [Coolify health checks](https://coolify.io/docs/applications/configuration/health-checks)
- [Coolify domains and internal ports](https://coolify.io/docs/core/networking/domains)

Every real user-reported misclassification should become a case here before you
change the prompt to fix it — that's how you prevent the fix from regressing later.
