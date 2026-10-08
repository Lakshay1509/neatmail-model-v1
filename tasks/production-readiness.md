# Production readiness review

Prepared for one server/container in Coolify. Experiment remains disabled by
default; no provider credentials or live email data were accessed during this pass.

Changes:

- Generic OpenAI failure responses prevent raw provider exceptions leaking email
  content or credentials to callers. Regression verified with a simulated leak.
- OpenRouter's documented transient `openrouter_in_flight_budget` 402 is retried
  within the existing two-attempt budget. Empty balance and key caps are not retried.
- Docker-owned Python health check uses the existing OpenAPI route, requires no
  curl/wget, and makes no paid inference calls.
- Five vulnerable dependency pins upgraded individually with all 29 tests passing
  after every upgrade: AnyIO 4.14.2, IDNA 3.15, pip 26.2.1, Starlette 1.3.1,
  urllib3 2.8.0. No other dependency pins were changed.
- GitHub Actions checks offline API/tests, compilation, dependency compatibility,
  pinned-package advisories, and the Docker build. Actions are pinned by commit SHA;
  token permissions are read-only, no deployment/provider secrets are used.

Verification:

- 29 offline tests pass. Synthetic report CLI verified in the implementation pass.
- Real Uvicorn process startup and exact Python health-check command passed on a
  local socket with mocked startup providers; no paid calls.
- `pip check`: no broken requirements.
- `pip-audit -r requirements.txt --no-deps --disable-pip`: no known vulnerabilities
  in the pinned packages. This is an advisory check, not a complete supply-chain
  guarantee; base-image OS packages require the deployment image scan.
- Compilation and `git diff --check` pass.

Deployment verification still required:

- Docker is unavailable in this local environment. The Docker build and Linux
  runtime need to pass CI/Coolify before deployment. CI has been added, not run remotely.
- Verify live account/model access and labeled email quality in staging.
- Verify both Coolify volume mounts survive container replacement and are writable;
  back them up. Maintain one deployment server.
- The shared dashboard key belongs to the trusted calling backend; do not expose
  it to browsers or mobile clients. That backend must authorize the supplied user ID.
- Review rate/size limits at your ingress according to expected traffic; the
  application is a trusted-backend API, not an anonymous public email gateway.
- Keep experiment configuration fixed during the measurement week. Review accuracy
  needs blinded labels; model confidence alone cannot establish correctness.
- Starlette still supports the current HTTPX test client with a deprecation warning.
  Treatment HTTPX runtime calls are unaffected; migrate the test client separately
  before Starlette removes that compatibility.

Upstream release notes reviewed:

- https://github.com/agronholm/anyio/blob/4.14.2/docs/versionhistory.rst
- https://github.com/kjd/idna/blob/master/HISTORY.md
- https://github.com/pypa/pip/blob/26.2.1/NEWS.rst
- https://github.com/encode/starlette/blob/1.3.1/docs/release-notes.md
- https://github.com/urllib3/urllib3/blob/2.8.0/CHANGES.rst
- https://openrouter.ai/docs/api/reference/limits#in-flight-spending-budget
- https://coolify.io/docs/applications/configuration/health-checks

Rollback: set `CLASSIFICATION_AB_ENABLED=false`, save, and restart/redeploy.
Preserve measurement/report volumes. GitHub CI does not automatically gate
Coolify's Git-triggered deployments: use a verified commit or configure the gate.
