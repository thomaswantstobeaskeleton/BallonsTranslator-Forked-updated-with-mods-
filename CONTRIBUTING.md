# Contributing to BallonsTranslatorPro

Guidelines for contributors and AI-assisted development, based on feedback from upstream maintainers.

---

## Ignored files

See **`.gitignore`** for the full list. Ignored paths include: config and cache dirs, `config.json`, `.env`, secrets, model weights (`*.pt`, `*.onnx`), IDE folders (`.vscode`, `.idea`, `.cursor`), and maintainer-only docs (e.g. `GITHUB_UPLOAD.md`, `COMMUNITY_RESPONSES.md`, `docs/CONNECT_GITHUB_AND_AUTO_PUSH.md`, `docs/PROMPT_FIND_MANGA_DOWNLOAD_SOURCES.md`, `RELEASE_NOTES.md`, `NOTES.md`, `docs/*_local.md`). Don’t commit these unless you’re a maintainer and intend to.

---

## Git commit practices

**Prefer many small, focused commits over few large ones.**

- Each commit should address **one logical change** (e.g. one feature, one fix, one refactor).
- Avoid commits like "various fixes" or "misc updates" that mix unrelated edits.
- This makes history easier to review, bisect, and merge into upstream as an experimental branch.

**Example:** Instead of one commit with "add OCR X, fix detector Y, update README", use three commits:
1. `Add OCR module X`
2. `Fix detector Y edge case`
3. `Update README installation section`

---

## Community etiquette

### Promoting this fork

- **Do not** write in every message or thread promoting this fork.
- **Do** ask upstream maintainers to mention this repository in their README. Suggested wording:
  > "There's a fork with more advanced features here: [link]"
- Let the maintainers decide how and where to link. One mention in their README is better than spamming links in every community message.

### Helping others with issues

When someone reports a problem that this fork solves:

- **Do** reply with something like: *"I solved this problem like this"* and a **link to the specific block of code** from this repository.
- **Do not** spam links to the fork in every reply.

**Response template:**
```
I solved this problem like this:

[Brief explanation of the fix.]

Here's the relevant code: [GitHub link to file#Lstart-Lend]

You can apply this patch or use the fork if that's easier.
```

**Link format for code blocks:** `https://github.com/OWNER/REPO/blob/BRANCH/path/file.py#L100-L120`

---

## Code quality and stability

**Avoid breaking code you didn't intend to change.**

- When adding or modifying a feature, **only touch the files and functions that need to change**.
- Do not refactor or "clean up" unrelated code in the same change.
- Test the modified behavior before committing.
- Upstream maintainers have limited time; random crashes in untouched functions are frustrating and hard to debug.

---

## UI translations (i18n)

Display languages live in `translate/<locale>.ts` (source) and `translate/<locale>.qm` (compiled,
loaded at runtime). Standard Qt widget strings ("Show details...", dialog buttons) come from Qt's
own `qtbase_<locale>.qm`, which the launcher installs automatically — they don't belong in our files.

Workflow:

1. `python scripts/update_translation.py` — refresh the `.ts` files from the source code.
2. Translate the `<translation type="unfinished">` entries (Qt Linguist, or any text editor).
3. `python scripts/strip_contaminated_translations.py --check` — verify no entry was translated
   into another language (the non-Chinese files used to be seeded from `zh_CN`, which showed a
   Chinese UI to everyone else). Drop `--check` to clear such entries.
4. `python scripts/compile_translation.py` — rebuild the `.qm` files, and commit them together
   with the `.ts` files.

Leave an entry empty/unfinished rather than guessing: Qt then falls back to the English source,
which is always better than text in a language the user did not choose.

---

## Upstream merge

This fork is suitable for merging into the main [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) repository as a **separate experimental branch**. If you are a maintainer and want to integrate these changes, consider merging into an `experimental` or `pro-features` branch rather than `main`, so users can opt in.
