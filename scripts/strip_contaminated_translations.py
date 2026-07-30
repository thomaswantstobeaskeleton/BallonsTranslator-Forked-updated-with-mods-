"""
Clear translations written in the wrong script from Qt .ts files.

The non-Chinese locale files were seeded from zh_CN, so most of their entries held
Chinese text: a user who picked, say, Korean got a UI full of Chinese (issue #153).
An entry in the wrong script is worse than no translation at all -- Qt falls back to
the English source when an entry is empty/unfinished -- so this script empties them
and marks them ``type="unfinished"`` for translators to fill in.

Usage:
    python scripts/strip_contaminated_translations.py [--check] [locale ...]

    --check   report contaminated entries and exit non-zero instead of rewriting.

Run ``python scripts/compile_translation.py`` afterwards to rebuild the .qm files.
"""
import argparse
import os.path as osp
import re
import sys
from glob import glob

MESSAGE_RE = re.compile(r'<message[^>]*>.*?</message>', re.S)
SOURCE_RE = re.compile(r'<source>(.*?)</source>', re.S)
TRANSLATION_RE = re.compile(r'<translation(?P<attrs>[^>]*)>(?P<text>.*?)</translation>', re.S)

# Locales that legitimately use Han characters.
HAN_LOCALE_PREFIXES = ('zh', 'ja', 'yue')


def has_han(text: str) -> bool:
    return any('一' <= ch <= '鿿' or '㐀' <= ch <= '䶿' for ch in text)


def is_contaminated(locale: str, source: str, translation: str, attrs: str) -> bool:
    """True when `translation` is written in a script the locale never uses."""
    if 'type=' in attrs and 'unfinished' in attrs:
        return False
    if not translation.strip():
        return False
    if locale.split('_')[0] in HAN_LOCALE_PREFIXES:
        return False
    return has_han(translation)


def process(path: str, check_only: bool = False):
    locale = osp.splitext(osp.basename(path))[0]
    text = open(path, encoding='utf-8').read()
    hits = []

    def replace_message(match):
        message = match.group(0)
        tr_match = TRANSLATION_RE.search(message)
        if tr_match is None:
            return message
        src_match = SOURCE_RE.search(message)
        source = src_match.group(1) if src_match else ''
        translation = tr_match.group('text')
        if not is_contaminated(locale, source, translation, tr_match.group('attrs')):
            return message
        hits.append((source, translation))
        if check_only:
            return message
        return message[:tr_match.start()] + '<translation type="unfinished"></translation>' + message[tr_match.end():]

    new_text = MESSAGE_RE.sub(replace_message, text)
    if hits and not check_only:
        open(path, 'w', encoding='utf-8', newline='').write(new_text)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('locales', nargs='*', help='locale codes to process (default: every .ts file)')
    parser.add_argument('--check', action='store_true', help='report only, do not rewrite files')
    args = parser.parse_args()

    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    trans_dir = osp.join(root, 'translate')
    if args.locales:
        paths = [osp.join(trans_dir, f'{code}.ts') for code in args.locales]
    else:
        paths = sorted(glob(osp.join(trans_dir, '*.ts')))

    total = 0
    for path in paths:
        if not osp.isfile(path):
            print(f'[WARN] {path} does not exist', file=sys.stderr)
            continue
        hits = process(path, args.check)
        total += len(hits)
        verb = 'contaminated' if args.check else 'cleared'
        print(f'{osp.basename(path)}: {len(hits)} {verb} entries')
        for source, translation in hits[:3]:
            print(f'    {source[:60]!r} -> {translation[:60]!r}')

    if args.check and total:
        print(f'\n{total} entries are translated into the wrong script.', file=sys.stderr)
        return 1
    if not args.check and total:
        print('\nRun "python scripts/compile_translation.py" to rebuild the .qm files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
