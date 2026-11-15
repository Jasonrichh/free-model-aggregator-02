# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Jasonrichh/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Jasonrichh/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one server that answers both the OpenAI Chat Completions API
and the Anthropic Messages API, in JSON and in server-sent events, with a real completion from
a free OpenRouter model behind every answer. Built for the
[make.fun bounty](https://make.fun/bounty/free-model-aggregator-02).

## How it works

- **One server, two wires.** `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages`
  (Anthropic), plus `GET /v1/models` in the shape each auth header names. An Anthropic request is
  translated to Chat Completions, dialled on the gateway, and translated back — including the
  streaming event order, stop reasons, usage, and tool calls.
- **Quality-ordered routing.** The brief's fourteen free models carry the house's quality rating.
  Each prompt's candidates are the models that meet the request's own requirements (tools, JSON
  mode, vision), ranked by rating, highest first. The best JSON-capable model serves the JSON
  prompts; the best model overall serves the rest.
- **Rate-limit-aware failover.** A gateway refusal (429, 5xx, 400, or an unreachable gateway)
  moves the prompt to the next-best candidate, up to 6 attempts, and every refusal is reported on
  the answer in the `x-makefun-upstream-failures` header. A model that refused once is not dialled
  again in the same trial. A 401 (no OpenRouter key pushed) ends the trial at once.
- **Wire guards.** Every upstream body names a `max_tokens` inside the gateway's 8192-token cap,
  and when a provider's answer omits usable usage, the server fills it from the text actually
  produced, so a served prompt is never lost to a missing usage count.
- **Catalog.** The fourteen free ids, once each — printed by `--list`, registered with the env,
  and listed by `GET /v1/models` on both auth headers.

## Run

```sh
# Print the catalog this server registers, byte for byte, and exit.
node main.ts --list

# Serve a trial: the env owns MAKEFUN_ENV_URL, the brief MAKEFUN_PROMPT_FILE,
# and the gateway OPENAI_BASE_URL / OPENAI_API_KEY.
node main.ts
```

Node 22 standard library only; no dependencies to install. `main.ts` runs directly
(`lib/providers.ts` reads the brief and builds the catalog, `lib/wire.ts` translates the two
wires and the SSE relays).

## Files

- `main.ts` — the server: brief, catalog, routes, router, failover, register, poll.
- `lib/providers.ts` — the brief's types and parser, the free model list, the catalog.
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation, the model field, the SSE writers.
- `LICENSE` — MIT.

## About

- Bounty: https://make.fun/bounty/free-model-aggregator-02
- Topics: `makefun`
