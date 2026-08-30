# Reasoning content: gap between inference-perf and the OTel GenAI spec

**Status:** analysis + proposed fix. Nothing in this document is implemented yet.

**Scope:** how reasoning ("thinking", chain-of-thought) is represented in OTel traces versus how
inference-perf's replay path represents it today, and what has to change to close the gap.

---

## 1. TL;DR

inference-perf models reasoning as a **message-level string field** (`reasoning_content`), which is
the OpenAI *wire* shape. The OTel GenAI conventions model it as a **part inside `parts[]`** with
`{"type": "reasoning", "content": "..."}`. These are two different places in the data structure.

The replay ingest path only understands `text`, `tool_call`, and `tool_call_response` parts. A
reasoning part is therefore **silently dropped** on the way in — and worse, in one code path it
causes a `KeyError`, and in another it silently degrades dependency-graph quality without any
error at all.

`wire_trace_converter.py` already **emits** the spec-correct reasoning part (default
`include_reasoning=True`). The replay loader deliberately passes `include_reasoning=False` today
precisely because the downstream path cannot consume it. **That `False` is the marker for where to
pick this work up.**

---

## 2. The correct format (target)

Source: OTel GenAI semantic conventions, non-normative examples —
<https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/non-normative/examples-llm-calls.md>

### 2.1 The reasoning part

```json
{
  "role": "assistant",
  "parts": [
    { "type": "reasoning", "content": "The user is asking about X, so I should ..." },
    { "type": "text",      "content": "The answer is 42." }
  ],
  "finish_reason": "stop"
}
```

Three properties matter and all three are load-bearing:

| Property | Value | Why it matters |
|---|---|---|
| Part type | `"reasoning"` | A **fifth** part type alongside `text` / `tool_call` / `tool_call_response`. |
| Content field | `"content"` (not `"text"`) | Consumers that special-case `part["text"]` will miss it. |
| Position | **First**, ahead of visible text | Reasoning precedes the answer. This breaks positional arithmetic — see §4.4. |

### 2.2 Where it appears

**Output messages only.** Reasoning shows up in `gen_ai.output.messages`, never in
`gen_ai.input.messages`. Verified empirically against the nm-output captures: all 41 responses
carry plaintext `reasoning`, and the replayed `messages[]` history in later turns contains none of
it. This matches the spec examples, which show reasoning solely on the output side.

Practical consequence: reasoning affects **output** token accounting and output-message matching.
It does not affect prefix/input reconstruction. That usefully narrows the blast radius.

### 2.3 Token accounting

```
gen_ai.usage.reasoning.output_tokens
```

Note two things:

1. The attribute name is `gen_ai.usage.reasoning.output_tokens` — **not** `gen_ai.usage.reasoning_tokens`.
   (An older conversion script in `inference-perf-ibm/lena_scripts/` used the wrong name.)
2. It is a **subset** of `gen_ai.usage.output_tokens`, not an addition to it. Do **not** sum them —
   that double-counts.

---

## 3. What inference-perf implements today

### 3.1 The wire/serving side — works, and is fine

`ChatMessage` at `inference_perf/apis/chat.py:171-176`:

```python
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    reasoning_content: Optional[str] = None      # <-- message-level string
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
```

`to_dict()` at `chat.py:189-202` emits **both** spellings for server compatibility:

```python
if self.reasoning_content is not None:
    msg["reasoning_content"] = self.reasoning_content
    msg["reasoning"] = self.reasoning_content
```

This is wired end to end and is **not** the problem:

- `replay_graph_session_datagen.py:562` — `reasoning_content=m.get("reasoning") or m.get("reasoning_content")`
- `replay_graph_session_datagen.py:933-986` — accumulates streaming reasoning deltas, folds
  `reasoning_text` into `output_text` for token counting
- `openai_client.py:316-340` — same dual-spelling read on both streaming and non-streaming responses
- `tests/test_reasoning_output_support.py` — existing coverage

**Keep all of this.** It is the correct representation for talking to an OpenAI-compatible server,
and it already handles the response direction properly. The gap is purely on **trace ingest**.

### 3.2 The ingest side — this is the gap

`gen_ai.output.messages` is parsed into `parts[]`, and every consumer of `parts[]` was written
against a closed set of three types. A reasoning part hits each of them differently.

---

## 4. The six consumer sites

All line numbers verified against branch `feat/load-wire-record-traces`.

### 4.1 `otel_trace_to_replay_graph.py:225-229` — silent drop

```python
for part in info["parts"]:
    part_type = part.get("type")
    if part_type == "text":
        text_parts.append(_coerce_text(part.get("content", "")))
    elif part_type == "tool_call":
        tool_calls.append(_normalize_tool_call(part))
    # no else -> a reasoning part vanishes here
```

Closed `if/elif` with no `else`. A reasoning part never reaches the replayed wire dict. **No warning
is logged**, so this is invisible in a run.

### 4.2 `otel_trace_utils.py:254-264` — hard `KeyError` ⚠️ **fix this first**

```python
for part in message_info["parts"]:
    if part["type"] == "tool_call":
        message_info["parts_text"].append(_format_tool_call(part))
    else:
        message_info["parts_text"].append(part["content"])   # <-- unguarded
```

Two separate bugs on adjacent lines:

- `part["type"]` — raw subscript, `KeyError` on any part without a `type`.
- `part["content"]` in the `else` branch — a reasoning part *does* have `content`, so it survives
  here by luck; but a **spec-shaped text part** `{"type": "text", "text": "..."}` (which uses `text`,
  not `content`) raises `KeyError: 'content'`.

This is the only site that crashes rather than degrading quietly, so it is the highest-priority fix
and should land regardless of the rest.

### 4.3 Token estimation — silently under-counts

- `otel_trace_utils.py:330-351` (`_extract_content_from_parts`): a 4-way `if/elif` chain over
  `text` / `tool_call` / `tool_result|tool_call_response` / `tool_use`, with **no** `else`. Reasoning
  contributes nothing.
- `otel_trace_utils.py:165-175` (`_extract_text_content`): the `parts` branch requires
  `part.get("type") == "text"`, so reasoning is skipped in output-token estimation too.

Effect: replayed output token counts come out **lower** than the source trace whenever reasoning is
present. Silent, and it skews benchmark results rather than failing.

Related, on the serving side: `chat.py:548-551` filters content blocks to
`part.get("type") in ["text", "input_text"]` when counting prompt tokens. Combined with
`replay_graph_session_datagen.py:1718-1727`, which JSON-stringifies unknown content blocks, an
unrecognized block can be **sent to the server but not counted locally**.

### 4.4 Positional arithmetic — silently degrades the dependency graph 🔥

This is the subtle one, and the reason a naive "just add reasoning to the parts list" change is
risky.

Three sites assume **"not a tool_call" implies "is the visible content"**:

```python
otel_trace_to_replay_graph.py:763   first_part_is_content = parts[0]["type"] != "tool_call"
otel_trace_to_replay_graph.py:814   first_part_is_content = parts[0]["type"] != "tool_call"
otel_trace_to_replay_graph.py:892   offset += 1 if part["type"] != "tool_call" else 2
```

Because a reasoning part is ordered **first** (§2.1), `parts[0]["type"] != "tool_call"` is `True` and
the reasoning part is **misclassified as the assistant's visible content**. The `offset` arithmetic
at line 892 then drifts, output-to-input message matching fails, and causal dependency edges
silently degrade to **TEMPORAL** edges.

The failure mode is the dangerous kind: the run completes, no error is raised, and the replay graph
is quietly less accurate than it looks. `otel_trace_to_replay_graph.py:820` —
`if not all(part["type"] == "tool_call" for part in parts[1:])` — is affected by the same shift.

### 4.5 Raw `part["type"]` subscripts

Seven sites use raw subscripting instead of `.get("type")`, each a latent `KeyError`:

```
otel_trace_utils.py:260
otel_trace_to_replay_graph.py:652, 674, 784, 785, 820, 892
```

### 4.6 `parts_text` / `raw_reconstructed_text` desync

`parts_text` (built in `reconstruct_each_part_in_message_info`, §4.2) is index-aligned with `parts`
and consumed positionally at `otel_trace_to_replay_graph.py:784-785`. Any fix that changes how many
entries `parts_text` gets — or their order — must keep it aligned with `parts`, or the
`parts_text[i]` lookups silently pick up the wrong text.

---

## 5. Validator

The repo's OTel validator allows only `text` / `tool_call` / `tool_call_response` and rejects
anything else as `output_message_unknown_part_type`. Converting all 65 nm-output files with
reasoning enabled produced that error on **every** file.

So the validator needs `reasoning` added to its allowed set as part of this work — otherwise
spec-correct traces fail validation.

---

## 6. Proposed fix

### Option A — normalize on ingest (recommended)

Fold the reasoning part into the existing `reasoning_content` field at the ingest boundary, and let
the rest of the pipeline stay exactly as it is.

Concretely, in `_replay_message_to_dict` (`otel_trace_to_replay_graph.py:225-229`):

```python
reasoning_parts: List[str] = []
for part in info["parts"]:
    part_type = part.get("type")
    if part_type == "text":
        text_parts.append(_coerce_text(part.get("content", "")))
    elif part_type == "tool_call":
        tool_calls.append(_normalize_tool_call(part))
    elif part_type == "reasoning":
        reasoning_parts.append(_coerce_text(part.get("content", "")))
if reasoning_parts:
    msg["reasoning_content"] = "\n".join(reasoning_parts)
```

Then **strip reasoning parts from `parts[]` before** the graph-matching code runs, so §4.4's
positional arithmetic keeps seeing exactly the `[content?, tool_call*]` shape it was written for.

**Why this is the right first move:**

- Reuses plumbing that already works end to end (§3.1) — `reasoning_content` is already sent,
  already streamed, already token-counted, already tested.
- **Zero risk to dependency-graph quality.** Reasoning never enters the positional arithmetic, so
  §4.4 cannot regress.
- Small and reviewable: one ingest site plus the defensive fixes below.
- Semantically sound. Reasoning is output-only (§2.2), so it never needed to participate in
  input-prefix matching in the first place.

**What it gives up:** reasoning is no longer a first-class part, so a future feature that needs to
know *where* in the sequence reasoning occurred (interleaved reasoning between tool calls, e.g.) is
not served. That is a real limitation, but nothing in inference-perf needs it today.

### Option B — plumb `reasoning` as a first-class fifth part type

Teach all six sites in §4 about `reasoning`, and rewrite the positional arithmetic at
`763 / 814 / 892` to filter reasoning out before computing offsets (or index by role rather than
position).

Correct and more future-proof, but it touches the most fragile code in the replay path — the
dependency-matching arithmetic — where failures are silent (edges quietly become TEMPORAL) rather
than loud. Do this only with dedicated test coverage asserting that dependency edges stay CAUSAL
when reasoning is present.

### Do these regardless of A or B

1. **`otel_trace_utils.py:260-263`** — guard both subscripts. This is a live `KeyError` today:
   ```python
   part_type = part.get("type")
   if part_type == "tool_call":
       message_info["parts_text"].append(_format_tool_call(part))
   else:
       message_info["parts_text"].append(_coerce_text(part.get("content") or part.get("text") or ""))
   ```
   Note this also fixes spec-shaped `{"type": "text", "text": ...}` parts, independent of reasoning.
2. **Replace the 7 raw `part["type"]` subscripts** (§4.5) with `.get("type")`.
3. **Add `reasoning` to the validator's allowed part types** (§5).
4. **Log, don't silently drop.** Add an `else` at `otel_trace_to_replay_graph.py:229` that
   `logger.debug`s unknown part types. The absence of this is why the gap went unnoticed.
5. **Token accounting** — include reasoning in output-token estimation (§4.3), and if
   `gen_ai.usage.reasoning.output_tokens` is read, remember it is a **subset** of `output_tokens`
   (§2.3). Do not add them.

### Flipping the switch

Once ingest handles reasoning, change `otel_trace_replay_datagen.py:300`:

```python
# today — reasoning omitted because the replay path cannot consume it
converted = convert_wire_file(trace_file, include_reasoning=False)
```

to drop the `include_reasoning=False` (the converter's default is already `True`). The comment above
that call documents exactly why the `False` is there; update it when you remove it.

No conversion-side work is needed: `wire_trace_converter.py` already emits spec-correct reasoning
for **both** APIs — Chat Completions (`message.reasoning` / `message.reasoning_content`, plaintext,
as Nemotron exposes it) and Responses (`reasoning` items' `summary[].summary_text`; note that
`encrypted_content` is unusable and is skipped).

---

## 7. Suggested sequencing

| # | Change | Risk | Notes |
|---|---|---|---|
| 1 | `otel_trace_utils.py:260-263` guards | Very low | Fixes a live `KeyError`; independent of reasoning. |
| 2 | 7 raw `part["type"]` → `.get("type")` | Very low | Pure hardening. |
| 3 | `else` + `logger.debug` for unknown part types | Very low | Makes future gaps visible. |
| 4 | Validator accepts `reasoning` | Low | Unblocks validating spec-correct traces. |
| 5 | Option A normalization + strip before matching | Low–medium | Needs the dependency-edge test below. |
| 6 | Token estimation includes reasoning | Medium | Changes reported numbers — baseline before/after. |
| 7 | Drop `include_reasoning=False` at datagen:300 | — | Only after 1-6. |

Steps 1-4 are safe to land on their own and are worth doing even if reasoning support is deferred.

### Tests worth writing

- Reasoning part → `reasoning_content` on the replayed wire message.
- **Dependency edges stay CAUSAL** (do not fall back to TEMPORAL) for an output message shaped
  `[reasoning, tool_call, tool_call]`. This is the §4.4 regression guard and the most valuable test
  here.
- `{"type": "text", "text": ...}` (spec spelling) does not raise `KeyError` in
  `reconstruct_each_part_in_message_info`.
- Output token estimation counts reasoning content.
- Validator accepts a trace containing a reasoning part.

Two tests named `test_chat_conversion_drops_reasoning` and
`test_responses_conversion_drops_reasoning_output` were **removed** from
`tests/test_wire_trace_converter.py` when the converter default became `include_reasoning=True` —
they asserted the old drop-by-default behavior. Worth re-adding as explicit
`include_reasoning=False` opt-out tests plus positive tests for the default.

---

## 8. Reference: current state of the converter

`inference_perf/datagen/replay/wire_trace_converter.py`, branch `feat/load-wire-record-traces`:

| Function | Reasoning behavior |
|---|---|
| `_reasoning_parts(reasoning)` | Emits `[{"type": "reasoning", "content": str}]`, or `[]`. |
| `_convert_chat_output(resp, include_reasoning=True)` | Prepends reasoning **before** text/tool_call. |
| `_convert_responses_output(items, finish_reason, include_reasoning=True)` | Handles `summary[].summary_text`; ordering falls out of `output[]` order. |
| `_build_span(..., reasoning_tokens)` | Sets `gen_ai.usage.reasoning.output_tokens` when non-zero. |
| `convert_wire_file(path, trace_id=None, include_reasoning=True)` | Default **on**; standalone CLI uses the default. |
| `otel_trace_replay_datagen.py:300` | Passes `include_reasoning=False` — the replay opt-out. |

Verified on a real nm-output capture: part order `['reasoning', 'tool_call', 'tool_call']`, field
name `content`, `finish_reason` at message level, and no reasoning in `gen_ai.input.messages`.

To produce spec-correct traces with reasoning for testing:

```bash
python -m inference_perf.datagen.replay.wire_trace_converter \
    --input /path/to/nm-output -o /path/to/nm-output-otel
```
