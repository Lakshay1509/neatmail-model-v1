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
- `/classify` and `/classify-batch` share one system prompt, so this validates both.
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

Every real user-reported misclassification should become a case here before you
change the prompt to fix it — that's how you prevent the fix from regressing later.
