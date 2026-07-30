"""Locale files must not contain translations in the wrong script (issue #153).

The non-Chinese .ts files were seeded from zh_CN, so picking e.g. Korean showed a
Chinese UI. Qt falls back to the English source for empty/unfinished entries, so an
untranslated entry is always better than one in a language the user did not pick.
"""
import os.path as osp
import sys
from glob import glob

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from scripts.strip_contaminated_translations import has_han, process  # noqa: E402

TRANSLATE_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'translate')


def test_has_han():
    assert has_han('批处理队列')
    assert not has_han('Batch processing queue')
    assert not has_han('일괄 처리 대기열')
    assert not has_han('Обработка очереди')


def test_no_locale_contains_chinese_translations():
    offenders = {}
    for path in sorted(glob(osp.join(TRANSLATE_DIR, '*.ts'))):
        hits = process(path, check_only=True)
        if hits:
            offenders[osp.basename(path)] = hits
    assert not offenders, (
        'These locales contain Chinese translations: '
        + ', '.join(f'{name} ({len(hits)})' for name, hits in offenders.items())
        + '. Run "python scripts/strip_contaminated_translations.py" then '
          '"python scripts/compile_translation.py".'
    )


def test_every_ts_file_has_a_compiled_qm():
    for path in sorted(glob(osp.join(TRANSLATE_DIR, '*.ts'))):
        qm = osp.splitext(path)[0] + '.qm'
        assert osp.isfile(qm), f'{osp.basename(qm)} is missing; run scripts/compile_translation.py'
        assert osp.getsize(qm) > 0, f'{osp.basename(qm)} is empty; run scripts/compile_translation.py'


def test_compiled_qm_files_are_up_to_date():
    """The shipped .qm must not still carry stripped Chinese entries."""
    pytest = __import__('pytest')
    QtCore = pytest.importorskip('qtpy.QtCore')

    for path in sorted(glob(osp.join(TRANSLATE_DIR, '*.ts'))):
        locale = osp.splitext(osp.basename(path))[0]
        if locale.split('_')[0] in ('zh', 'ja', 'yue'):
            continue
        translator = QtCore.QTranslator()
        assert translator.load(locale, TRANSLATE_DIR), f'could not load {locale}.qm'
        for context, source in (
            ('BatchQueueDialog', 'Batch processing queue'),
            ('MainWindow', 'Save project'),
        ):
            translated = translator.translate(context, source)
            assert not has_han(translated or ''), (
                f'{locale}.qm still translates {source!r} to Chinese; '
                'run scripts/compile_translation.py after stripping the .ts files'
            )
