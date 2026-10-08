# Implementation checklist

- [x] Stable assignment, settings, private measurement writer and usage normalization.
- [x] Jev policy and conditional summary requests with response validation.
- [x] Single/batch routing, fallback, and shared control batch accounting.
- [x] Weekly Markdown/JSON report and report tests.
- [x] Coolify configuration/rollback instructions and final review.

Verification: 27 offline tests passed, including mocked API requests. The actual
report CLI produced inspected Markdown/JSON from synthetic data in
`reports/synthetic-demo/`. Inverting the empty-tags gate was caught by its
regression test; the mutation was restored immediately.

Operator checks before enabling: live model/account access, representative labeled
email evaluation, and Coolify volume persistence across container replacement.
No deployment, paid provider calls, or real experiment results were produced.

Final production pass: 29 tests pass, patched dependency audit reports no known
vulnerabilities, and Uvicorn/health-check runtime verification passes with mocks.
See `production-readiness.md` for the final review and deployment limitations.
