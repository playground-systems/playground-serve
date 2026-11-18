# Security

## Reporting a vulnerability

Please report it privately, not in a public issue or pull request. Use GitHub's private vulnerability reporting: the
repository's **Security** tab, then **Report a vulnerability**. The report and the discussion of a fix stay private
between you and the maintainers until an advisory is published.

If that button is not there, open an issue that asks the maintainers for a private way to report, and leave every
detail of the problem out of it.

Helpful in a report: what is affected (file, command or option), how to reproduce it, what an attacker needs and gains,
and the commit you tested. Leave out real API keys and tokens; a placeholder is enough.

## Supported versions

Playground Serve is alpha (0.1.1). Fixes go to `main`; there are no release branches.

## In scope

- The direct port (`pgs up --direct`): the per-boot token and the TLS certificate that the client pins
  (`src/playground_serve/worker/direct.py`, `src/playground_serve/client/client.py`, [docs/WIRE.md](docs/WIRE.md)).
  For example, reaching a worker's server without the token, or making a client accept another certificate.
- Key handling in `pgs` and the client: how the Runpod API key is read, saved (`~/.config/playground-serve/key`) and
  sent ([docs/USAGE.md](docs/USAGE.md#the-api-key)), and how entry secrets such as `HF_TOKEN` reach only the endpoint's
  environment. For example, the key or a token showing up in output, logs, state files or a request to anyone but
  Runpod.
- The worker bundle: the scripts that `pgs up` packs into the endpoint's environment (`RS_FILES`, built by
  `src/playground_serve/bundle.py` from `worker/` and `families/`), what they download and run when a worker starts,
  and what the worker exposes.

## Out of scope

Runpod's platform and gateway, and the models and their code that a worker downloads (openpi, MolmoAct2 and the
others listed in [NOTICE](NOTICE)): please report those to their own projects.
