// The trial brief, the free-model list, and the catalog document.
//
// The brief (the file MAKEFUN_PROMPT_FILE points at) is prose with two JSON documents embedded
// in it, each tagged by a `schema` string under this competition's prefix:
//
//   - the model list (`free-model-aggregator-02/models/v1`): one row per free model, keyed by the
//     gateway id (`openrouter/<vendor>/<model>:free`), with the house rating in milli-points, the
//     context length, and whether the model takes tools, images, and JSON mode;
//   - the prompt manifest (`free-model-aggregator-02/prompts/v1`): the six prompts the client will
//     send, in order, each with its wire, its streaming flag, its weight, and what a model must be
//     able to do to serve it.
//
// This module pulls those two documents out of the prose, turns them into rows the router can
// rank, and builds the one catalog document the server registers and `--list` prints. Only free
// models are ever routed or listed. Everything is a pure function of the brief file: no clock,
// no randomness, so `--list` and the register body agree byte for byte on a live pass and on its
// replay.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** The prefix every schema string in this competition's brief carries. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The provider's public endpoint: the catalog names it, never the gateway's trial address. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The env var the entrant's OpenRouter key is pushed under; it never enters the sandbox. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;

/** Whether a value is a plain JSON object (not null, not an array). */
export function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as read. `error` names what could not be read, if anything. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/**
 * Scan `text` for top-level JSON objects that carry a string `schema`, in the order they appear.
 * Braces inside strings are skipped, so a schema string in prose or a nested object never fools
 * the scan; a `{` that does not open a parseable object is passed over.
 */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  let i = 0;
  while (i < text.length) {
    const open = text.indexOf('{', i);
    if (open === -1) break;
    const close = matchingClose(text, open);
    if (close === -1) break;
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text.slice(open, close + 1));
    } catch {
      parsed = null;
    }
    if (isRecord(parsed) && typeof parsed.schema === 'string') found.push(parsed);
    i = open + 1;
  }
  return found;
}

/**
 * The index of the `}` that closes the object opening at `start`, or -1 when the text ends first.
 * String literals are walked as literals: a brace or an escape inside one never counts.
 */
function matchingClose(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** A value as a string, or '' when it is not one. */
function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/** A value as a finite number, or null when it is not one. */
function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** A value as a boolean, or null when it is not one. */
function asBoolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this competition's prefix that holds one of `keys` as an array, and
 * which key it was. A brief that spells its lists under the sibling contest's names still reads.
 */
function findArrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row
 * that gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the
 * gateway spells it (`openrouter/vendor/model:free`). The brief lists free models, so a row is
 * free unless it says `free: false`; a paid row is never routed.
 */
function toModelRow(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = asString(v.id) !== '' ? asString(v.id) : asString(v.model);
  if (raw === '') return null;
  const named = asString(v.provider) !== '' ? asString(v.provider) : asString(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  const rating = asNumberOrNull(v.rating_milli);
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: asString(v.display_name) !== '' ? asString(v.display_name) : raw,
    rating_milli: rating === null ? 0 : rating,
    context_length: asNumberOrNull(v.context_length),
    tools: asBoolOrNull(v.tools),
    vision: asBoolOrNull(v.vision),
    json: asBoolOrNull(v.json),
    free: v.free !== false,
  };
}

/** One prompt row. A row with no id, or no requirements object, still reads (with no needs). */
function toPromptRow(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const minContext = asNumberOrNull(requires.min_context);
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(minContext !== null ? { min_context: minContext } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this competition's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = findArrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models: ModelRow[] = [];
  for (const v of modelsDoc.doc[modelsDoc.key] as unknown[]) {
    const m = toModelRow(v);
    if (m !== null) models.push(m);
  }
  const promptsDoc = findArrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts: PromptRow[] = [];
  for (const v of promptsDoc.doc[promptsDoc.key] as unknown[]) {
    const p = toPromptRow(v);
    if (p !== null) prompts.push(p);
  }
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves, each once.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local address: the
 * gateway's origin differs between a live pass and its replay, and the catalog is part of the
 * state export a replay must reproduce byte for byte.
 *
 * This lists every free model in the brief — exactly the fourteen dialable ids the catalog column
 * scores — so keep it that way: register this set and keep GET /v1/models (both auth headers)
 * exactly that set, since listing fewer loses the column as surely as listing more.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
