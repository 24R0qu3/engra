# engra Backlog

Feedback-driven issues prioritised by impact. Source: user evaluation on a Doxygen HTML corpus.

---

## #1 HTML chunking destroys semantic structure [DONE]

**Problem:** `read_html` dumps the entire page as a single `Section`, which is then mechanically
sliced at 1500-char boundaries. On Doxygen pages this splits function signatures from their
doc comments and mixes unrelated API surface into the same chunk.

**Fix:** Split by h1–h4 headings (same pattern as `read_markdown`) using BeautifulSoup tree
traversal so each section stays intact. Fallback to single-section for heading-free HTML.

**Files:** `src/engra/readers.py` — `read_html`

---

## #2 Score distribution is collapsed (0.83–0.87 range)

**Problem:** `intfloat/multilingual-e5-large` compresses cosine similarity into a narrow band on
technical docs. All results land near 0.84–0.87 regardless of relevance, making ranking
meaningless and eroding user trust.

**Options (ascending effort):**
1. Relative score normalisation within each result set (cosmetic, zero deps).
2. Cross-encoder re-ranking after vector retrieval (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).
   New optional dependency; adds meaningful spread and improves top-1 accuracy.
3. Switch embedding model to one better calibrated for code/technical text.

**Files:** `src/engra/commands.py` — `_data_search`

---

## #3 No handling of stub/TBD patterns [DONE]

**Problem:** Chunks containing `TBD`, `TODO`, `FIXME`, unimplemented stubs, or
`raise NotImplementedError` carry high signal but are invisible in ranked results. The
most valuable find (a TBD stub) only surfaced after a very targeted query.

**Fix:** Post-processing pass in `_data_search` adds a `notable: bool` flag to each result
dict. `cmd_search` renders a `[!]` marker next to flagged results.

**Files:** `src/engra/commands.py` — `_data_search`, `cmd_search`

---

## #4 Cross-file relationship surfacing

**Problem:** Doxygen HTML contains hyperlinks between class pages, member docs, and interface
specs. engra ignores these, missing the opportunity to surface "this function is stubbed here,
the interface is defined there" relationships proactively.

**Proposed approach:**
- During `read_html` indexing, extract `<a href>` links between pages and store as metadata.
- At search time, after retrieving top-K results, do a secondary lookup for chunks from
  linked files.
- Requires a metadata schema addition but `chromadb`'s `where` filter makes lookups cheap.

**Files:** `src/engra/readers.py` — `read_html`; `src/engra/commands.py` — `_data_search`
