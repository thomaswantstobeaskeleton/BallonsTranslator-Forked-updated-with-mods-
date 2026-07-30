# Translation context and terminology consistency

This document describes the design and implementation of **cross-page translation context** and **glossary-based terminology consistency** so that long works (e.g. cultivation manhua like *Rebirth of the Urban Immortal Cultivator*) keep terms and style consistent across all images/chapters.

---

## 1. Problem

- **Current behavior:** Each page (or batch of text blocks) is translated in isolation. The translator sees only the current page’s source text.
- **Consequences:**
  - **Terminology drift:** The same concept is translated differently on different pages (e.g. 丹田 → "dantian" on one page, "core" on another; 真气 → "true qi" vs "spiritual energy").
  - **Style drift:** Tone and register can change from page to page.
  - **Broken cohesion:** Character names, place names, and domain terms (cultivation, techniques, etc.) should stay fixed across the whole project.

---

## 2. Research summary

### 2.1 Glossary-based consistency

- **Industry practice:** Translation glossaries are key–value lists (source term → approved translation). They are injected into the prompt or used as a constraint so the model uses the same wording every time.
- **Effect:** Studies and vendors report that without a glossary, terminology inconsistency in chunked long-form translation can be **15–30%**; with a glossary it drops to **below 5%**.
- **Use cases:** Brand names, technical terms, named entities, and **domain vocabulary** (e.g. cultivation terms: 金丹, 元婴, 真气, 丹田).

### 2.2 Context window (previous content)

- **Idea:** When translating page *N*, give the model a short “recent context” from pages *N−k* … *N−1* (e.g. last 1–3 pages) as source + translation.
- **Benefits:**
  - The model sees how terms and style were chosen earlier and tends to repeat them.
  - Improves narrative flow and register consistency.
- **Caveats:**
  - Consumes part of the context window; need a cap (e.g. last 2 pages, or last ~500 words).
  - Only works when pages are translated in **reading order** (sequential pipeline or ordered queue).

### 2.3 Document-level translation (full-doc)

- **Idea:** Translate the entire work as one long document, then map segments back to pages.
- **Pros:** Maximum consistency.
- **Cons:** Token limits, memory, and the current app design (page-by-page pipeline) make this a larger change. Not implemented here; left as a future option.

### 2.4 Combined approach (chosen)

- **Glossary:** User- and/or project-defined term list (source → translation). Injected into every LLM call so the same terms are used on every page.
- **Previous-page context:** When translating in order, append a compact “recent pages” context (e.g. last 1–2 pages’ source + translation) to the prompt so style and ad-hoc terms stay in line.

---

## 3. Design

### 3.1 Glossary

- **Where it lives:**
  - **Translator param (LLM_API_Translator):** Multiline field `translation_glossary`. Format: one entry per line, e.g. `source -> target` or `source = target` (same as existing `keyword_replacements`).
  - **Project-level (optional):** Project JSON can store `translation_glossary` (list of `{source, target}` or a string). Loaded when the project is opened and **merged** with the translator’s own glossary for the run. Allows a “Rebirth of the Urban Immortal Cultivator” glossary to travel with the project.
- **How it’s used:** Before each request, the effective glossary (translator + project) is formatted into a short block and appended to the **system prompt** (or the user prompt), e.g.  
  `Use these exact translations for terms: 丹田 -> dantian; 真气 -> true qi; 金丹 -> golden core.`  
  The model is instructed to prefer these when they appear in the source.

### 3.2 Previous-page context

- **When:** Only when translation runs **sequentially** (one page after another in reading order). In the current code this is either:
  - Sequential pipeline (no parallel translate), or
  - Parallel translate thread that still processes pages in queue order (so we can treat “already finished” pages as “previous”).
- **What we pass:** For the current page index `i`, take the last **N** pages (e.g. N=1 or 2) with indices `i−N … i−1`. For each of those pages we have:
  - Source texts (from blocks)
  - Translations (already filled)
  We format them into a single “Previous context (for terminology and style consistency):” section with a few lines per page to avoid blowing the context window.
- **Where it’s used:** Appended to the **user prompt** (or a dedicated “context” message) so the model sees recent source and translation and can match style/terms.

### 3.3 Integration points

- **Base translator:**  
  - `translate_textblk_lst(textblk_lst, **kwargs)` can accept optional kwargs.  
  - Optional method `set_translation_context(context)` so the pipeline can set “previous pages” and/or “project glossary” before calling `translate_textblk_lst` for the current page.
- **LLM_API_Translator:**
  - New params: `translation_glossary` (multiline), `context_previous_pages` (number 0–5), optional `series_context_prompt` (e.g. “This is a cultivation manhua. Keep terms consistent.”).
  - `set_translation_context(previous_pages_data)` to receive the list of (sources, translations) for previous pages.
  - In `_assemble_prompts`: inject glossary into system prompt; inject previous-page context into user prompt (with a token budget).
- **Pipeline (module_manager):**
  - When about to translate a page, compute ordered page list and current index; collect last N pages’ source+translation; call `translator.set_translation_context(...)` then `translate_textblk_lst(blk_list)`.
  - For **parallel** translate: when a page is popped from the queue, “previous” = all pages with index &lt; current. Only pages that have already been translated can be used (so we only add context for pages already in `finished_counter`). So we pass context only when we’re past the first few pages and only for pages that are already done. That matches “previous pages” in reading order.

### 3.4 Project glossary (save/load)

- In `ProjImgTrans`:
  - `translation_glossary`: list of `{"source": str, "target": str}` or similar. Default `[]`.
  - `to_dict()`: include `translation_glossary` in the saved JSON.
  - `load_from_dict()`: if key exists, load `translation_glossary`; else leave empty.
- When running the pipeline, the effective glossary = translator param glossary + project glossary (merged, no duplicates by source).

---

## 4. User-facing settings (LLM_API_Translator)

| Parameter | Type | Description |
|------------|------|-------------|
| **translation_glossary** | Multiline | Entries `source -> target` (or `=`, `:`) for terms that must stay consistent. Used on every page. |
| **context_previous_pages** | Number (0–5) | How many immediately preceding pages to include as context (source + translation). 0 = off. |
| **series_context_prompt** | Short text | Optional. E.g. “This is a cultivation manhua. Use consistent terms for cultivation and character names.” Prepended or merged into system prompt. |

Project-level glossary is stored in the project JSON and merged at run time; it can be edited in a future UI (e.g. project settings or a “Translation glossary” tab).

### 4.1 Cache-friendly page history (Config → Translator → LLM context)

The settings above render context **inside the prompt text**. A second, complementary mode sends
previously translated pages as **real conversation turns** with a token-budgeted rolling window,
provider prompt-cache reuse, and context-length recovery. It is documented separately in
[`doc/LLM_TRANSLATION_CONTEXT.md`](../doc/LLM_TRANSLATION_CONTEXT.md); when it is enabled, the
in-prompt page context above is suppressed so context is not sent twice.

---

## 5. References

- IntlPull / context windows and chunking for translation (2026).
- Translation glossary definition and practices (Lara Translate, Google Cloud Translate).
- LLM-BT and terminology standardization (arXiv).
- Document-level translation and discourse (AIModels.fyi).

---

## 7. File changes (implementation)

- `utils/series_context_store.py`: Series folder helpers – `get_series_context_dir`, `load_series_glossary`, `load_recent_context`, `append_page_to_series_context`, `merge_glossary_no_dupes`.
- `utils/proj_imgtrans.py`: Add `translation_glossary` and **series_context_path** to project; save/load in `to_dict` / `load_from_dict`.
- `modules/translators/base.py`: Optional `set_translation_context(..., series_context_path=...)` and **append_page_to_series_context** (no-op in base).
- `modules/translators/trans_llm_api.py`: Params (glossary, context_previous_pages, series_context_prompt, **series_context_path**); load series glossary and recent context from folder; merge into prompt; **append_page_to_series_context** writes to store.
- `ui/module_manager.py`: Before translating a page, pass **series_context_path** (from project or translator) into `set_translation_context`; after each successful translate, call **append_page_to_series_context** with the page’s sources and translations.
