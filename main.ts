// free-model-aggregator-01 entry: one server, two client wires, ten providers on three wires.
//
// Built on the kit's starter: the brief, the catalog probe, the register and the /work poll are
// the starter's. What this entry adds, against the starter's marked weaknesses:
//   - quality: every request is routed to the highest-rated free model that meets its
//     requirements (the quality table's rating_milli), ties broken by the brief's provider
//     order, then by model id: a pure function of the brief, so a replay routes identically.
//   - failover: a provider call that answers 402, 429 or 5xx, or fails outright, is counted
//     and the request moves to the next candidate on a different provider. A provider that
//     fails three calls in a row is left alone for the rest of the trial (a spent budget or
//     an outage refuses everything anyway), and one whose refusal names its budget is left
//     alone at once.
//   - wires: all three upstream wires are spoken. The OpenAI Chat Completions shape is the
//     canonical middle: an Anthropic client is translated into it (the starter's wire.ts),
//     an Anthropic or Gemini upstream is translated out of it and its answer back (lib/wire2.ts),
//     so the system prompt, the user text, the tool names, max_tokens, the stop sequences, the
//     images and the JSON mode reach the provider in its own wire and the provider's answer,
//     its usage and its stop reason reach the client in the client's wire.
//   - one limit kept from the echoes: an Anthropic-wire provider cannot report JSON mode back,
//     so a request the client put in JSON mode is never routed to one.
//
// Everything is deterministic: no clock and no randomness anywhere; generated ids derive from
// the request's place in the workload.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicStreamToOpenAI,
  anthropicToOpenAIResponse,
  geminiStreamToOpenAI,
  geminiToOpenAIResponse,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  streamFromText,
} from './lib/wire2.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 25000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A provider failing this many calls in a row is treated as spent for the rest of the trial. */
const DEAD_AFTER_FAILURES = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
/** Consecutive failed inference calls per provider id, reset by any success. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerOrder: number;
}

/** Every free model meeting the requirements, best first: rating, then brief order, then id. */
function candidates(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerOrder) => {
    if (!probe.ok) return;
    if (jsonMode && probe.provider.wire === 'anthropic') return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, providerOrder });
    }
  });
  out.sort((a, b) => b.rating - a.rating || a.providerOrder - b.providerOrder || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

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

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: string;
}

/** The upstream call for one candidate, in the provider's own wire, from the canonical body. */
function upstreamCall(candidate: Candidate, canonical: Json): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    if (canonical.stream === true) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers, body: JSON.stringify({ ...canonical, model: candidate.model.id }) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (canonical.stream === true) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers, body: JSON.stringify(openAIToAnthropicRequest(canonical, candidate.model.id)) };
  }
  const verb = canonical.stream === true ? 'streamGenerateContent' : 'generateContent';
  if (canonical.stream === true) headers.accept = 'text/event-stream';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}`, headers, body: JSON.stringify(openAIToGeminiRequest(canonical)) };
}

/**
 * One request, served: try the candidates in order until one answers. Returns the successful
 * candidate and the upstream body text, or null when every candidate refused.
 */
async function serve(canonical: Json, requires: Requirement, jsonMode: boolean): Promise<{ candidate: Candidate; text: string } | null> {
  const ranked = candidates(requires, jsonMode);
  const attempted = new Set<string>();
  const noteFailure = (candidate: Candidate, detail: string): boolean => {
    const budgetSpent = detail.toLowerCase().includes('budget');
    const count = (consecutiveFailures.get(candidate.probe.provider.id) ?? 0) + 1;
    consecutiveFailures.set(candidate.probe.provider.id, budgetSpent ? 99 : count);
    return budgetSpent;
  };
  for (const pass of [0, 1]) {
    for (const candidate of ranked) {
      const providerId = candidate.probe.provider.id;
      if (attempted.has(providerId)) continue;
      const dead = (consecutiveFailures.get(providerId) ?? 0) >= DEAD_AFTER_FAILURES;
      if (dead && pass === 0) continue;
      attempted.add(providerId);
      const call = upstreamCall(candidate, canonical);
      // A refusal on the provider's own failure schedule refuses one call number and the next
      // call is a fresh draw, so one refusal earns one immediate retry on the same model; a
      // refusal that names a spent budget earns none, and the provider is done for the trial.
      for (let attempt = 0; attempt < 2; attempt++) {
        let upstream: Response;
        try {
          upstream = await fetch(call.url, { method: 'POST', headers: call.headers, body: call.body, signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
        } catch (err) {
          noteFailure(candidate, (err as Error).message);
          break;
        }
        const text = await upstream.text();
        if (!upstream.ok) {
          if (noteFailure(candidate, text)) break;
          continue;
        }
        consecutiveFailures.set(providerId, 0);
        return { candidate, text };
      }
    }
  }
  return null;
}

/** The canonical (OpenAI-shaped) answer of one successful upstream call: JSON object or SSE text. */
function canonicalAnswer(candidate: Candidate, text: string, stream: boolean, idSeed: number): { json: Json } | { sse: string } {
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') {
    return stream ? { sse: text } : { json: JSON.parse(text) as Json };
  }
  if (wire === 'anthropic') {
    return stream
      ? { sse: anthropicStreamToOpenAI(text, candidate.model.id, idSeed) }
      : { json: anthropicToOpenAIResponse(JSON.parse(text) as Json, candidate.model.id, idSeed) };
  }
  return stream
    ? { sse: geminiStreamToOpenAI(text, candidate.model.id, idSeed) }
    : { json: geminiToOpenAIResponse(JSON.parse(text) as Json, candidate.model.id, idSeed) };
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

/** Serve one client request end to end, on either client wire. */
async function answer(req: IncomingMessage, res: ServerResponse, clientWire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const idSeed = manifestIndex;
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';

  // The canonical body: the client's own OpenAI body, or its Anthropic body translated.
  const canonical: Json = clientWire === 'openai' ? { ...body } : anthropicToOpenAI(body, 'routed');
  if (stream) {
    canonical.stream = true;
    canonical.stream_options = { include_usage: true };
  }
  const jsonMode =
    isRecord(canonical.response_format) && (canonical.response_format.type === 'json_object' || canonical.response_format.type === 'json_schema');

  const served = await serve(canonical, requires, jsonMode);
  if (served === null) {
    return sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_error', 'every free provider refused this request');
  }
  const { candidate, text } = served;

  // Same-wire fast path for Anthropic: the provider already speaks the client's wire.
  if (clientWire === 'anthropic' && candidate.probe.provider.wire === 'anthropic') {
    if (stream) return sendSse(res, text);
    return sendJson(res, 200, JSON.parse(text) as Json);
  }

  const answerValue = canonicalAnswer(candidate, text, stream, idSeed);
  if (clientWire === 'openai') {
    if ('sse' in answerValue) return sendSse(res, answerValue.sse);
    return sendJson(res, 200, answerValue.json);
  }
  // Anthropic client, canonical answer.
  if ('sse' in answerValue) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for await (const event of anthropicEventsFromOpenAI(streamFromText(answerValue.sse), requested)) res.write(event);
    res.end();
    return;
  }
  return sendJson(res, 200, openAIToAnthropic(answerValue.json, requested));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await answer(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await answer(req, res, 'anthropic');
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

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

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
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
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
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
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
