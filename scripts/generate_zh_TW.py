"""
Generate the Traditional Chinese (zh_TW) translation from the Simplified one (zh_CN).

zh_CN is the most complete catalog, so zh_TW is derived from it with OpenCC's
``s2twp`` profile (Simplified -> Traditional with Taiwan vocabulary) instead of being
translated from scratch. Entries already translated by hand in zh_TW.ts are kept.

Usage:
    python scripts/generate_zh_TW.py [--check]

    --check   report how many entries would change and exit non-zero if any do.

Run ``python scripts/compile_translation.py zh_TW`` afterwards to rebuild the .qm file.
"""
import argparse
import os.path as osp
import re
import sys

MESSAGE_RE = re.compile(r'<message[^>]*>.*?</message>', re.S)
SOURCE_RE = re.compile(r'<source>(.*?)</source>', re.S)
TRANSLATION_RE = re.compile(r'<translation(?P<attrs>[^>]*)>(?P<text>.*?)</translation>', re.S)
CONTEXT_NAME_RE = re.compile(r'<name>(.*?)</name>', re.S)


def _converter():
    try:
        import opencc
    except ImportError:
        print('opencc is required: pip install opencc-python-reimplemented', file=sys.stderr)
        raise SystemExit(2)
    return opencc.OpenCC('s2twp')


def _existing_translations(path: str) -> dict:
    """Map (context, source) -> translation for hand written entries."""
    if not osp.isfile(path):
        return {}
    text = open(path, encoding='utf-8').read()
    result = {}
    for context in re.findall(r'<context>.*?</context>', text, re.S):
        name_match = CONTEXT_NAME_RE.search(context)
        name = name_match.group(1) if name_match else ''
        for message in MESSAGE_RE.findall(context):
            src = SOURCE_RE.search(message)
            tr = TRANSLATION_RE.search(message)
            if src is None or tr is None:
                continue
            if 'unfinished' in tr.group('attrs') or not tr.group('text').strip():
                continue
            result[(name, src.group(1))] = tr.group('text')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='report only, do not write zh_TW.ts')
    args = parser.parse_args()

    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    src_path = osp.join(root, 'translate', 'zh_CN.ts')
    dst_path = osp.join(root, 'translate', 'zh_TW.ts')
    if not osp.isfile(src_path):
        print(f'{src_path} does not exist', file=sys.stderr)
        return 2

    convert = _converter().convert
    manual = _existing_translations(dst_path)
    text = open(src_path, encoding='utf-8').read()
    converted = generated = kept = 0

    def replace_context(context_match):
        context = context_match.group(0)
        name_match = CONTEXT_NAME_RE.search(context)
        name = name_match.group(1) if name_match else ''

        def replace_message(match):
            nonlocal converted, generated, kept
            message = match.group(0)
            tr_match = TRANSLATION_RE.search(message)
            src_match = SOURCE_RE.search(message)
            if tr_match is None or src_match is None:
                return message
            source = src_match.group(1)
            translation = tr_match.group('text')
            if (name, source) in manual:
                kept += 1
                new_translation = manual[(name, source)]
                attrs = ''
            elif not translation.strip() or 'unfinished' in tr_match.group('attrs'):
                return message
            else:
                new_translation = convert(translation)
                generated += 1
                if new_translation != translation:
                    converted += 1
                attrs = ''
            return (message[:tr_match.start()]
                    + f'<translation{attrs}>{new_translation}</translation>'
                    + message[tr_match.end():])

        return MESSAGE_RE.sub(replace_message, context)

    out = re.sub(r'<context>.*?</context>', replace_context, text, flags=re.S)
    out = out.replace('<TS version="2.1" language="zh_CN"', '<TS version="2.1" language="zh_TW"')
    out = re.sub(r'(<TS[^>]*?)language="zh_CN"', r'\1language="zh_TW"', out)

    print(f'zh_TW: {generated} generated ({converted} changed by conversion), {kept} kept from manual edits')
    if args.check:
        existing = open(dst_path, encoding='utf-8').read() if osp.isfile(dst_path) else ''
        if existing != out:
            print('zh_TW.ts is out of date; run python scripts/generate_zh_TW.py', file=sys.stderr)
            return 1
        return 0

    open(dst_path, 'w', encoding='utf-8', newline='').write(out)
    print(f'Wrote {dst_path}')
    print('Run "python scripts/compile_translation.py zh_TW" to rebuild the .qm file.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
