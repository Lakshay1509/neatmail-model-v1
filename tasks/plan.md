# Classification experiment implementation

Implement the approved SPEC-classification-ab.md without deployment or live traffic changes.

1. Stable configuration/assignment and private cost-ledger writer. Verify offline boundary tests.
2. Jev decisions and conditional OpenRouter summaries. Verify policy, response validation and retries with mock HTTP.
3. Integrate single and mixed-user batch routing with existing OpenAI functions. Verify API contracts with mocked startup services.
4. Offline weekly report and Coolify operations docs. Verify aggregation with synthetic events and generate both artifacts.
5. Review correctness/privacy/accounting; run full offline suite and compilation.

Dependencies: 1 → 2 → 3 → 4 → 5. No new runtime dependencies. Tests use unittest.
Runtime provider access, semantic calibration and Coolify persistence require the operator's configured accounts/server.
