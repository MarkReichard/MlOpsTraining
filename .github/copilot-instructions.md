# Copilot Coding Instructions

## Zero Trust
- Never trust input — validate and sanitize at every boundary (API, DB, external service).
- Apply least-privilege: request only the permissions and data a component actually needs.
- Authenticate and authorize every call, even internal service-to-service calls.
- Treat secrets as ephemeral; never hardcode credentials, tokens, or keys.

## Clean Code
- Keep functions short and single-purpose — if it needs a comment to explain *what* it does, split it.
- Minimize cognitive load: limit nesting, early-return on guard clauses, avoid boolean flags as parameters.
- No magic numbers or strings — use named constants.

## Naming
- Names must read like plain English: `getUserByEmail`, `isTokenExpired`, `OrderProcessingService`.
- Avoid abbreviations, single-letter variables (except loop indices), and generic names (`data`, `info`, `temp`).
- Boolean names must be questions: `isActive`, `hasPermission`, `canRetry`.

## Reuse
- Search for an existing utility, library, or pattern before writing new code.
- Extract shared logic immediately when it appears a second time.
- Prefer composition over duplication.

## Code Quality
- Fix all linter and Sonar issues before considering code complete.
- Never suppress or disable a scan rule to clear a finding — fix the root cause.
- Address warnings with the same priority as errors.
