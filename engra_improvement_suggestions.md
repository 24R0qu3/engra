# engra MCP — Improvement Suggestions

Based on practical use during a session querying a Doxygen-generated API reference (ISOcores v1.19.2).

---

## 1. Negative Query Support

The most significant gap: when asked "does X exist?", the tool always returns the closest semantic match rather than signalling "not found." This produces false positives and forces a follow-up manual verification step.

**Suggested fix:** Introduce a confidence threshold below which the tool responds with an explicit "no relevant results found" instead of a low-confidence match. Optionally expose the score to the caller so they can decide how to proceed.

---

## 2. Chunk Boundaries Anchored to Declarations

Doxygen comment blocks and their corresponding declarations were frequently split across chunk boundaries. A method's `@param` / `@return` documentation would appear without the actual function signature, or vice versa.

**Suggested fix:** Anchor chunk splits at the declaration level. A chunk must always contain both the doc comment and the complete declaration it belongs to, including all parameters and the return type.

---

## 3. Class Context in Every Chunk

Returned chunks often lacked the enclosing class name. A snippet showing `virtual void RemoveDevice(...)` is ambiguous without knowing it lives in `TcAppInterface`.

**Suggested fix:** Prepend each chunk with a fully-qualified breadcrumb:

```
namespace::ClassName::methodName
```

This makes every result self-contained and removes the need to infer context from surrounding text.

---

## 4. Structured "List" Query Mode

For questions like "is there a callback for event X?", semantic similarity is a poor fit. The more useful operation is: retrieve all members of a class, then let the caller determine what is absent.

**Suggested fix:** Add a structured query mode — e.g. `list all pure virtual methods in TcAppInterface` — that returns a complete, structured member list rather than a ranked similarity result. This is particularly valuable for confirming absence.

---

## 5. Cross-Reference Metadata

Doxygen generates explicit cross-references (`@see`, `\callgraph`, documented call sites in `@note`). These were among the most valuable pieces of information encountered (e.g. discovering that `RemoveDevice()` is also called from `OpenTaskDataResource()`), but surfaced only when the exact chunk happened to be retrieved by chance.

**Suggested fix:** When a chunk is returned, attach any explicitly documented cross-references as structured metadata alongside the content. Example:

```json
{
  "content": "...",
  "cross_references": [
    "TC_ISOXML::XmlRawResourceAccess::OpenTaskDataResource",
    "TC::Tc::Stop",
    "TC::Tc::ProcessAll"
  ]
}
```

---

## 6. Atomic Enum Chunks

When searching for an enum type (e.g. `TcWorkingSetMgrWarningType`), only one or two values were returned because the enum body was split across chunks. This makes exhaustive checks ("are there any values related to X?") unreliable.

**Suggested fix:** Treat each enum definition as a single atomic, non-splittable chunk regardless of its length. An enum's value list is only meaningful as a whole.

---

---

## 7. Sentence-Aware Chunking

The current `chunk_text()` splits at a fixed character position (1500 chars), regardless of word or sentence boundaries. This means chunks can start or end mid-word or mid-sentence, giving the embedding model syntactically broken input at every seam. For code, a function signature can be severed from its first parameter.

**Suggested fix:** Split at sentence or paragraph boundaries up to a maximum size, falling back to the nearest word boundary. This eliminates broken input at chunk edges and improves the quality of the embedding for each chunk.

---

## 8. Underused Model Context Window

`multilingual-e5-large` handles up to 512 tokens. At ~4 chars/token, the current `CHUNK_SIZE = 1500` chars ≈ 375 tokens — leaving ~130 tokens unused per chunk. Fewer, larger chunks mean fewer seams and better per-chunk coherence.

**Suggested fix:** Increase `CHUNK_SIZE` to ~1800–2000 chars to saturate the model's context window more fully. Combine with fix #7 so the larger window is filled with complete sentences.

---

## 9. Variable Shadowing in Embed Loop

In `_data_index` (`commands.py`), the embed loop uses `chunk` as the loop variable, silently shadowing the `chunk` variable from the outer section-chunking loop:

```python
for chunk, meta, embedding in zip(chunk_texts, chunk_metas, embed_gen):
    documents.append(chunk)  # 'chunk' here is doc_text, not the raw chunk
```

This works correctly because `chunk_texts` already contains the breadcrumb-prefixed versions, but the shadowing makes the code hard to follow and is a future maintenance risk.

**Suggested fix:** Rename the embed-loop variable to `doc_text` to match the variable it was built from.

---

---

## 10. Embedding Batch Size Too Conservative

`model.embed(chunk_texts, batch_size=32)` uses a batch size of 32. ONNX Runtime vectorises across the batch — on CPU, 64–128 is typically 20–40% faster with no quality change. The current value was likely a safe default rather than a measured optimum.

**Suggested fix:** Increase default `batch_size` to 64. Make it configurable via `config.toml` (`embedding.batch_size`) so users with more RAM can push it higher.

---

## 11. ONNX Runtime Thread Count Not Configured

`load_model()` passes no thread configuration, so fastembed/ONNX Runtime defaults to however many threads it picks internally (often 1–2). This leaves most CPU cores idle during inference.

**Suggested fix:** Pass `threads=os.cpu_count()` (or a config-controlled value) to `TextEmbedding(...)`. This parallelises the matrix operations inside each inference call across all available cores.

---

## 12. File Reading and Embedding Are Fully Serial

For directory indexing, the loop is: read file → parse → chunk → embed → read next file. File reading and HTML/PDF parsing are IO + CPU work that could be done on a background thread while the main thread runs ONNX inference on the previous file's chunks. This hides parsing latency almost entirely.

**Suggested fix:** Use a `ThreadPoolExecutor` with one worker to pre-load and parse the next file's sections while the current file is being embedded (producer/consumer pattern). Requires extracting the read+chunk step from `_data_index` into a separate stage.

---

## 13. No GPU / Accelerated ONNX Provider Support

fastembed supports alternative ONNX execution providers (`CUDAExecutionProvider`, `ROCMExecutionProvider`, etc.) that can be 5–10× faster than CPU. Currently the provider is hardcoded to CPU by default with no way to opt in.

**Suggested fix:** Add an `embedding.provider` config key (default `"cpu"`). When set to `"cuda"` or `"rocm"`, pass the appropriate providers list to `TextEmbedding(...)`. Fall back to CPU gracefully if the requested provider is unavailable.

---

## Summary

| # | Issue | Impact | Fix Complexity | Status |
|---|---|---|---|---|
| 1 | No "not found" response | High — causes false positives | Low | ✅ Done |
| 2 | Doc/declaration split across chunks | High — incomplete results | Medium | ✅ Done |
| 3 | Missing class context in chunks | Medium — ambiguous results | Low | ✅ Done |
| 4 | No structured list query | Medium — poor for absence checks | Medium | ✅ Done |
| 5 | Cross-references not surfaced | Medium — misses key relationships | Medium | ✅ Done |
| 6 | Enums split across chunks | Low-Medium — incomplete enum lists | Low | ✅ Done |
| 7 | Character-based chunking cuts mid-sentence | High — broken embedding input | Low | ✅ Done |
| 8 | Chunk size underuses model context window | Medium — suboptimal retrieval | Low | ✅ Done |
| 9 | Variable shadowing in embed loop | Low — code clarity/maintenance risk | Trivial | ✅ Done |
| 10 | Batch size too conservative | Medium — 20–40% speed gain available | Trivial | ✅ Done |
| 11 | ONNX thread count not configured | Medium — CPU cores left idle | Trivial | ✅ Done |
| 12 | File reading and embedding fully serial | Medium — parsing latency wasted | Medium | ✅ Done |
| 13 | No GPU/accelerated provider support | High (if GPU available) | Low | ✅ Done |
