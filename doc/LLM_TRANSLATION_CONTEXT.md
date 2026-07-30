# LLM translation context and glossary

Configure in **Config → Translator → LLM context** (applies to the `LLM_API_Translator`).

## Prior context

| Setting | Meaning |
|---|---|
| **Current page only** (default) | Each request contains only the current page. |
| **Previously translated pages** | Previously translated pages of the project are sent as prior conversation turns. |

With *Previously translated pages*, each finished page is replayed as a `user` message
(the same JSON request that page used) followed by the `assistant` reply that produced its
translations. The model therefore sees real examples of how earlier pages were handled, which
keeps character names, honorifics, terminology, and tone consistent across a chapter.

The message prefix is deliberately kept byte-identical between adjacent pages, so providers
that cache prompts (OpenAI, DeepSeek, Anthropic-compatible gateways, …) bill the shared part at
the cached rate. Run with debug logging to see whether the prefix is being reused:

```
LLM token usage: page=012.png, prompt=8102, completion=311, total=8413, cache_hit=7936
```

### Token budget

`Token budget` caps how much history is sent. Pages are whole units: when the budget is reached,
the oldest pages are dropped in bulk down to 60% of the budget, so the following pages can grow
the window again without invalidating the cache on every request.

If a provider still rejects the request as too long, the oldest pages are dropped and the request
is retried — the current page and glossary are never truncated.

### When history is rebuilt

History is only reused when it is provably still valid. It is rebuilt from the project when:

- the project was (re)loaded, or a different project is open;
- the model, source language, system prompt, or token budget changed;
- the page is not the one right after the last translated page;
- a page kept in the window was edited since it was rendered;
- the previous page is not fully translated, or was translated into another target language.

Pages translated into a different target language are never used as history: the target language
is recorded per page when a translation finishes.

## Glossary

`Glossary file` accepts three formats, chosen by extension:

```jsonc
// glossary.json
[{"src": "勇者", "dst": "Hero", "info": "protagonist's title"}]
```

```
# glossary.txt   ("source->translation #optional note")
勇者->Hero #protagonist's title
```

```
# glossary.tsv   (source <TAB> translation <TAB> optional note)
勇者	Hero	protagonist's title
```

Blank lines and lines starting with `#`, `//`, or `\\` are ignored. Duplicate rows are dropped;
a source mapped to two different translations is reported as an error instead of being applied
silently. The file is re-read only when it changes on disk.

| Mode | Behavior |
|---|---|
| **Matching terms only** (default) | Only entries whose source appears on the current page are attached to that request. |
| **Whole glossary** | Every entry is sent once, before the history, as part of the stable cached prefix. |

The glossary constrains wording only; it cannot change the target language, the item ids, the
item count, or the response format.

## Notes

- This is independent of the project-level *Translation context (project)* dialog (series path,
  project glossary) and of the translator's own `context_previous_pages_count` parameter — when
  *Previously translated pages* is selected, the older in-prompt page context is suppressed so
  context is not sent twice.
- Token counts are estimated with `tiktoken` when it is installed, and with a deterministic
  fallback otherwise.
