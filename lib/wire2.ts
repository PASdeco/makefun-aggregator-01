// The upstream wires the starter did not speak: Anthropic and Gemini as upstreams, with the
// OpenAI Chat Completions shape as the canonical middle. Every function here is a pure function
// of its arguments: no clock, no randomness, so a live pass and its replay produce the same bytes.
// Deterministic ids come from the upstream's own id where it has one, else from the request's
// index in the workload, passed in by the caller.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** The fixed epoch second the simulated services stamp on completions; mirrored for our own. */
const FIXED_CREATED = 1767225600;

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// OpenAI (canonical) -> Anthropic request
// ---------------------------------------------------------------------------

/** One OpenAI content part as an Anthropic content block; null for parts Anthropic cannot carry. */
function anthropicBlock(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
    if (data !== null && data[2] === ';base64') {
      return { type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[3] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      const content = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length === 1 && blocks[0].type === 'text' ? text : blocks });
      continue;
    }
    // user
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    const blocks = (Array.isArray(raw.content) ? raw.content : []).flatMap((p) => {
      const b = anthropicBlock(p);
      return b === null ? [] : [b];
    });
    messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') out.stop_sequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) out.stop_sequences = stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Anthropic -> OpenAI (canonical) response
// ---------------------------------------------------------------------------

/** Anthropic's stop_reason as OpenAI's finish_reason (same stop family). */
export function anthropicFinish(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object naming `model`. */
export function anthropicToOpenAIResponse(body: Json, model: string, idSeed: number): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = num(usage.input_tokens);
  const output = num(usage.output_tokens);
  const id = typeof body.id === 'string' && body.id !== '' ? `chatcmpl-${body.id.replace(/^msg_/, '')}` : `chatcmpl-agg${idSeed}`;
  return {
    id,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: anthropicFinish(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

/** The frames of one SSE body: each frame's event name (null when unnamed) and data payload. */
export function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const rawFrame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (rawFrame.trim() === '') continue;
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of rawFrame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    frames.push({ event, data: dataLines.join('\n') });
  }
  return frames;
}

function openAIChunk(id: string, model: string, delta: Json, finish: string | null): string {
  return sseEvent(null, {
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
}

function usageChunk(id: string, model: string, input: number, output: number): string {
  return sseEvent(null, {
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  });
}

/** An Anthropic SSE stream's text as one OpenAI SSE stream (a single string of frames). */
export function anthropicStreamToOpenAI(text: string, model: string, idSeed: number): string {
  let id = `chatcmpl-agg${idSeed}`;
  let input = 0;
  let output = 0;
  let stop: unknown = 'end_turn';
  const out: string[] = [];
  let started = false;
  const toolIndexByBlock = new Map<number, number>();
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message)) {
      if (typeof data.message.id === 'string' && data.message.id !== '') id = `chatcmpl-${data.message.id.replace(/^msg_/, '')}`;
      if (isRecord(data.message.usage)) input = num(data.message.usage.input_tokens);
      if (!started) {
        started = true;
        out.push(openAIChunk(id, model, { role: 'assistant', content: '' }, null));
      }
      continue;
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const blockIndex = num(data.index);
      const toolIndex = toolIndexByBlock.size;
      toolIndexByBlock.set(blockIndex, toolIndex);
      out.push(
        openAIChunk(
          id,
          model,
          {
            tool_calls: [
              {
                index: toolIndex,
                id: typeof data.content_block.id === 'string' ? data.content_block.id : `call_${toolIndex}`,
                type: 'function',
                function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' },
              },
            ],
          },
          null,
        ),
      );
      continue;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        out.push(openAIChunk(id, model, { content: data.delta.text }, null));
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const toolIndex = toolIndexByBlock.get(num(data.index)) ?? 0;
        out.push(
          openAIChunk(id, model, { tool_calls: [{ index: toolIndex, function: { arguments: data.delta.partial_json } }] }, null),
        );
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta)) stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = num(data.usage.output_tokens);
      continue;
    }
  }
  if (!started) out.push(openAIChunk(id, model, { role: 'assistant', content: '' }, null));
  out.push(openAIChunk(id, model, {}, anthropicFinish(stop)));
  out.push(usageChunk(id, model, input, output));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

// ---------------------------------------------------------------------------
// OpenAI (canonical) -> Gemini request
// ---------------------------------------------------------------------------

/** A base64 data URL split into its media type and data; null when the URL is not one. */
function dataUrl(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (match === null || match[2] !== ';base64') return null;
  return { media: match[1] ?? 'image/png', data: match[3] };
}

/** An OpenAI Chat Completions request as a Gemini generateContent request for `model`. */
export function openAIToGeminiRequest(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNameById = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      const name = toolNameById.get(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '') ?? 'tool';
      const content = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: content } } }] });
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const parsed = dataUrl(part.image_url.url);
          if (parsed !== null) parts.push({ inlineData: { mimeType: parsed.media, data: parsed.data } });
          else parts.push({ fileData: { fileUri: part.image_url.url } });
        }
      }
    }
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      const name = typeof call.function.name === 'string' ? call.function.name : '';
      if (typeof call.id === 'string') toolNameById.set(call.id, name);
      parts.push({ functionCall: { name, args: argsObject(call.function.arguments) } });
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') config.stopSequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) config.stopSequences = stop.filter((s) => typeof s === 'string');
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Gemini -> OpenAI (canonical) response
// ---------------------------------------------------------------------------

/** Gemini's finishReason as OpenAI's finish_reason (same stop family). */
export function geminiFinish(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT') return 'content_filter';
  return 'stop';
}

/** The text and tool calls of one Gemini candidate's content. */
function geminiParts(content: unknown): { text: string; toolCalls: Json[] } {
  let text = '';
  const toolCalls: Json[] = [];
  const parts = isRecord(content) && Array.isArray(content.parts) ? content.parts : [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  return { text, toolCalls };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAIResponse(body: Json, model: string, idSeed: number): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const { text, toolCalls } = geminiParts(candidate.content);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = num(usage.promptTokenCount);
  const output = num(usage.candidatesTokenCount);
  return {
    id: `chatcmpl-agg${idSeed}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: geminiFinish(candidate.finishReason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini SSE stream's text as one OpenAI SSE stream (a single string of frames). */
export function geminiStreamToOpenAI(text: string, model: string, idSeed: number): string {
  const id = `chatcmpl-agg${idSeed}`;
  const out: string[] = [openAIChunk(id, model, { role: 'assistant', content: '' }, null)];
  let input = 0;
  let output = 0;
  let finish: string | null = null;
  for (const frame of sseFrames(text)) {
    if (frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const { text: piece } = geminiParts(candidate.content);
      if (piece !== '') out.push(openAIChunk(id, model, { content: piece }, null));
      if (typeof candidate.finishReason === 'string') finish = geminiFinish(candidate.finishReason);
    }
    if (isRecord(data.usageMetadata)) {
      input = num(data.usageMetadata.promptTokenCount);
      output = num(data.usageMetadata.candidatesTokenCount);
    }
  }
  out.push(openAIChunk(id, model, {}, finish ?? 'stop'));
  out.push(usageChunk(id, model, input, output));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A string of SSE frames as a ReadableStream of its bytes, for the starter's stream translators. */
export function streamFromText(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}
