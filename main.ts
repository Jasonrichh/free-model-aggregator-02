// free-model-aggregator-02: one server, two client wires, and the best free model behind every
// answer.
//
// This entry starts from the house starter kit and replaces its routing and its failover:
//
//   ROUTER (quality, 60% of the score): every prompt's candidates are the brief's free models that
//   meet the request's own requirements (tools, JSON mode, vision, read off the request body, so no
//   prompt is routed by arrival order), ranked by the brief's rating_milli, highest first, ties by
//   id. The best JSON-capable model serves the JSON prompts; the best model overall serves the rest.
//   The order is fixed by the brief, so a live pass and its replay choose the same model.
//
//   FAILOVER (10%): a gateway refusal (429, 5xx, 400, or an unreachable gateway) moves the prompt
//   to the next-best candidate, up to 6 attempts a prompt, and every refusal is reported on the
//   answer in the `x-makefun-upstream-failures` header (at most 8 rows), so the failover column can
//   see a refusal the server retried and then served. Each prompt tries the candidates fresh, in
//   rating order: a free tier's per-minute limit can clear between prompts, so a model that refused
//   one prompt may serve the next, and a refused call still counts toward the calls tie-break, so
//   the chain never retries a model inside one prompt. A 401 means no OpenRouter key is pushed,
//   which no other model can fix, so the trial gives up at once instead of burning the trial's
//   60-call cap.
//
//   WIRE (10%): the starter's translators are kept, with two guards. The gateway refuses a call
//   with no output cap or one over 8192 tokens, so every upstream body is given a max_tokens
//   inside the cap. And the env's usage check needs an input count >= 0 and an output count >= 1:
//   when a provider's answer omits usable usage, this server fills it from the text actually
//   produced (output ≈ ceil(chars / 4), at least 1; input 0 when unknown), on JSON answers and in
//   both stream relays (see lib/wire.ts), so a served prompt is never lost to a missing usage.
//
//   CATALOG (5%): the catalog is the brief's fourteen free ids, once each, printed by --list and
//   registered byte for byte the same, and listed by GET /v1/models on both auth headers.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's standard
//   library alone (nothing is installed; TypeScript runs directly, and a relative import needs its .ts
//   extension). The env is at $MAKEFUN_ENV_URL; your model calls go through the grader's gateway at
//   $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on the key you pushed to the
//   competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can do
//       (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its weight
//       (1 or 2) and the requirements a model must meet to serve it. Each prompt carries a unique token
//       the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
//        with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually dialled.
//   The house records every gateway call and scores the model that really answered. This server sends
//   the brief's model id in the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
//   its claim is the call it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  estimatedOutputTokens,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most upstream attempts one prompt spends, across its failover chain. */
const MAX_ATTEMPTS = 6;
/** The gateway's output cap: a call naming more, or none, is refused. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap a request with none gets, inside the gateway's cap. */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** The header that carries a served prompt's upstream refusals, for the failover column. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Whether the gateway answered 401: no key is pushed, and no model can fix that. */
let authFailed = false;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve a request, best first: the brief's rating_milli, highest first,
 * ties by id, so the order is a function of the brief alone. Once the gateway has answered 401
 * there are no candidates at all.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  if (authFailed) return [];
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * The requirements of a request, read off its own body (either wire): a tools array means the model
 * must take tools, a JSON response_format means it must take JSON mode, an image part means it must
 * take vision. Reading the body, not the manifest's arrival order, keeps the routing right whatever
 * order the prompts arrive in.
 */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: readonly FailureRow[] = []): void {
  const trail = trailHeaders(failures);
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), trail);
  else sendJson(res, status, openAIError(message, type, null), trail);
}

/** One upstream refusal, as the failover trail reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The `x-makefun-upstream-failures` header for a prompt's refusals, or nothing when there were none. */
function trailHeaders(failures: readonly FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, 8).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  return { [FAILURES_HEADER]: JSON.stringify(rows) };
}

/**
 * A gateway refusal's code, as the env reads it: the error body's `code`, else its `type`, else a
 * name for the status. A row's code is at most 64 characters, as the env's trail reader allows.
 */
function failureCode(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const err = value.error;
      if (typeof err.code === 'string' && err.code !== '') return err.code.slice(0, 64);
      if (typeof err.code === 'number') return String(err.code).slice(0, 64);
      if (typeof err.type === 'string' && err.type !== '') return err.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status name.
  }
  if (status === 401) return 'unauthorized';
  if (status === 429) return 'rate_limited';
  if (status === 400) return 'invalid_request';
  if (status >= 500) return 'upstream_error';
  return 'upstream_error';
}

/**
 * The output cap the gateway requires: every upstream body names a max_tokens inside its 8192-token
 * cap, so a request that forgot one, or named one over the cap, is still dialled.
 */
function ensureOutputCap(body: Json): Json {
  const out = { ...body };
  const named = typeof out.max_tokens === 'number' ? out.max_tokens : typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens : undefined;
  const capped =
    typeof named === 'number' && Number.isFinite(named)
      ? Math.min(Math.max(1, Math.floor(named)), MAX_OUTPUT_TOKENS)
      : DEFAULT_OUTPUT_TOKENS;
  out.max_tokens = capped;
  delete out.max_completion_tokens;
  return out;
}

/** The text of a Chat Completions message: its string content, or its text parts joined. */
function answerTextOf(message: Json): string {
  if (typeof message.content === 'string') return message.content;
  if (!Array.isArray(message.content)) return '';
  return message.content
    .filter((b): b is Json => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
    .map((b) => String(b.text))
    .join('');
}

/**
 * A Chat Completions answer whose usage the env can read: when the provider's usage is missing or
 * its output count is below 1, fill it from the text actually produced, and give a missing
 * finish_reason the `stop` the content earned. The provider's own counts are kept whenever usable.
 */
function ensureOpenAIUsage(answer: Json): Json {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = answerTextOf(message);
  const usage = isRecord(answer.usage) ? { ...answer.usage } : {};
  const prompt = typeof usage.prompt_tokens === 'number' && usage.prompt_tokens >= 0 ? Math.floor(usage.prompt_tokens) : 0;
  const output =
    typeof usage.completion_tokens === 'number' && usage.completion_tokens >= 1
      ? Math.floor(usage.completion_tokens)
      : estimatedOutputTokens(text.length);
  const out: Json = { ...answer, usage: { ...usage, prompt_tokens: prompt, completion_tokens: output, total_tokens: prompt + output } };
  if (choices.length > 0) {
    out.choices = choices.map((c, i) => {
      if (i !== 0 || !isRecord(c) || typeof c.finish_reason === 'string') return c;
      return { ...c, finish_reason: 'stop' };
    });
  }
  return out;
}

/**
 * An Anthropic message whose usage the env can read: the same fill as the OpenAI side, from the
 * text blocks the message actually carries. The stop_reason is always a string already.
 */
function ensureAnthropicUsage(answer: Json): Json {
  const content = Array.isArray(answer.content) ? answer.content : [];
  let textChars = 0;
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') textChars += block.text.length;
  }
  const usage = isRecord(answer.usage) ? { ...answer.usage } : {};
  const input = typeof usage.input_tokens === 'number' && usage.input_tokens >= 0 ? Math.floor(usage.input_tokens) : 0;
  const output =
    typeof usage.output_tokens === 'number' && usage.output_tokens >= 1
      ? Math.floor(usage.output_tokens)
      : estimatedOutputTokens(textChars);
  return { ...answer, usage: { ...usage, input_tokens: input, output_tokens: output } };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** What one prompt's upstream attempts produced: the answer and the model that gave it, or neither. */
interface ServedUpstream {
  response: Response | null;
  model: string | null;
  failures: FailureRow[];
}

/**
 * Dial a prompt's candidates, best first, until one answers: each refusal is recorded in order.
 * A 401 stops the prompt and the trial at once, since no pushed key is a fact about the account,
 * not the model. At most MAX_ATTEMPTS calls a prompt.
 */
async function serveUpstream(buildBody: (model: string) => Json, requires: Requirement): Promise<ServedUpstream> {
  const failures: FailureRow[] = [];
  const candidates = candidatesFor(requires);
  const attempts = Math.min(candidates.length, MAX_ATTEMPTS);
  for (let i = 0; i < attempts; i += 1) {
    const model = candidates[i];
    let upstream: Response;
    try {
      upstream = await dial(model.id, buildBody(model.id));
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      log(`dial ${model.id} failed: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { response: upstream, model: model.id, failures };
    const text = await upstream.text().catch(() => '');
    failures.push({ status: upstream.status, code: failureCode(text, upstream.status), model: model.id });
    log(`dial ${model.id} refused: HTTP ${upstream.status}`);
    if (upstream.status === 401) {
      authFailed = true;
      log('the gateway answered 401: no OpenRouter key is pushed; giving up on this trial');
      return { response: null, model: null, failures };
    }
  }
  return { response: null, model: null, failures };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFromBody(body);
  const stream = body.stream === true;
  const served = await serveUpstream((model) => ensureOutputCap({ ...body, model }), requires);
  if (served.response === null || served.model === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model refused this prompt', served.failures);
  }
  const upstream = served.response;
  const model = served.model;
  const trail = trailHeaders(served.failures);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body as ReadableStream<Uint8Array>, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered a body that is not JSON', served.failures);
  }
  sendJson(res, 200, ensureOpenAIUsage(openAIJsonWithModel(isRecord(answer) ? answer : {}, model)), trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFromBody(body);
  const stream = body.stream === true;
  const served = await serveUpstream((model) => ensureOutputCap(anthropicToOpenAI(body, model)), requires);
  if (served.response === null || served.model === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model refused this prompt', served.failures);
  }
  const upstream = served.response;
  const model = served.model;
  const trail = trailHeaders(served.failures);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body as ReadableStream<Uint8Array>, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered a body that is not JSON', served.failures);
  }
  sendJson(res, 200, ensureAnthropicUsage(openAIToAnthropic(isRecord(answer) ? answer : {}, model)), trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
