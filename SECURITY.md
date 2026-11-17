# Security Policy

## Supported versions

Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](../../security/advisories/new) for this
repository rather than opening a public issue. Include steps to reproduce and
the affected endpoint or module. You can expect an initial response within a
few days.

## Deployment notes

- The service has **no authentication** and is intended for local use. Do not
  expose it to untrusted networks without putting it behind an authenticating
  proxy.
- No API keys or cloud credentials are required. Keep any local `.env` file out
  of version control (it is ignored by `.gitignore`).
- `/chat` forwards prompts to the Ollama instance at `OLLAMA_BASE_URL`; point it
  only at an Ollama server you trust.
