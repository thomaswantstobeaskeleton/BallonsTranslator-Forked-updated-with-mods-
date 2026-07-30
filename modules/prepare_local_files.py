from typing import Union, List, Optional, Dict, Any
from datetime import datetime, timezone
import os.path as osp
import os

from . import INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS
from .base import BaseModule, LOGGER
import utils.shared as shared
from utils.download_util import download_and_check_files, check_local_file

# Optional ONNX inpainter models (modules only registered when onnxruntime is installed)
OPTIONAL_ONNX_MODELS = [
    {
        "url": "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
        "files": osp.join(shared.PROGRAM_PATH, "data/models/inpainting_lama_2025jan.onnx"),
    },
    {
        "url": "https://huggingface.co/mayocream/lama-manga-onnx/resolve/main/lama-manga.onnx",
        "files": osp.join(shared.PROGRAM_PATH, "data/models/lama_manga.onnx"),
    },
]


def download_and_check_module_files(module_class_list: Optional[List[BaseModule]] = None):
    if module_class_list is None:
        module_class_list = []
        for registered in [INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS]:
            for module_key in registered.module_dict.keys():
                module_class_list.append(registered.get(module_key))

    for module_class in module_class_list:
        if module_class.download_file_on_load or module_class.download_file_list is None:
            continue
        for download_kwargs in module_class.download_file_list:
            all_successful = download_and_check_files(**download_kwargs)
            if all_successful:
                continue
            LOGGER.error(f'Please save these files manually to specified path and restart the application, otherwise {module_class} will be unavailable.')


def download_optional_onnx_models():
    """Download optional ONNX inpainter assets (called when optional_onnx package is selected)."""
    for kw in OPTIONAL_ONNX_MODELS:
        download_and_check_files(**kw)


def maybe_quantize_optional_onnx_models():
    """
    Optional post-download quantization for ONNX inpainter models.
    Enable with env:
      BT_QUANTIZE_OPTIONAL_ONNX=1
    Optional mode:
      BT_ONNX_QUANT_MODE=dynamic|static (default: dynamic)
    """
    if str(os.getenv("BT_QUANTIZE_OPTIONAL_ONNX", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return

    mode = str(os.getenv("BT_ONNX_QUANT_MODE", "dynamic")).strip().lower()
    try:
        from scripts.quantize_optional_onnx import quantize_optional_onnx_models
    except Exception as e:
        LOGGER.warning("Optional ONNX quantization unavailable: %s", e)
        return
    try:
        summary = quantize_optional_onnx_models(mode=mode)
        LOGGER.info("Optional ONNX quantization complete: %s", summary)
    except Exception as e:
        LOGGER.warning("Optional ONNX quantization failed (continuing without quantized models): %s", e)


def prepare_pkuseg():
    try:
        import pkuseg
    except ImportError:
        import spacy_pkuseg as pkuseg

    flist = [
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/postag.zip',
            'files': ['features.pkl', 'weights.npz'],
            'sha256_pre_calculated': ['17d734c186a0f6e76d15f4990e766a00eed5f72bea099575df23677435ee749d', '2bbd53b366be82a1becedb4d29f76296b36ad7560b6a8c85d54054900336d59a'],
            'archived_files': 'postag.zip',
            'save_dir': 'data/models/pkuseg/postag'
        },
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/spacy_ontonotes.zip',
            'files': ['features.msgpack', 'weights.npz'],
            'sha256_pre_calculated': ['fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18', '5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e'],
            'archived_files': 'spacy_ontonotes.zip',
            'save_dir': 'data/models/pkuseg/spacy_ontonotes'
        },
    ]
    for files_download_kwargs in flist:
        download_and_check_files(**files_download_kwargs)

    PKUSEG_HOME = osp.join(shared.PROGRAM_PATH, 'data/models/pkuseg')
    pkuseg.config.pkuseg_home = PKUSEG_HOME

    # there must be data/models/pkuseg/postag.zip and data/models/pkuseg/spacy_ontonotes.zip
    # otherwise the dumb package download these models again becuz its dumb checking
    p = osp.join(PKUSEG_HOME, 'postag.zip')
    if not osp.exists(p):
        os.makedirs(p)

    p = osp.join(PKUSEG_HOME, 'spacy_ontonotes.zip')
    if not osp.exists(p):
        os.makedirs(p)


def prepare_local_files_forall():
    """
    Download model files for selected packages only.
    model_packages_enabled is None = legacy "download all". Otherwise only modules in selected packages.
    """
    import utils.config as program_config
    from utils.model_packages import (
        get_module_classes_for_packages,
        package_ids_include_pkuseg,
        package_ids_include_optional_onnx,
    )

    pcfg = program_config.pcfg
    package_ids = getattr(pcfg, "model_packages_enabled", None)

    module_list = get_module_classes_for_packages(package_ids)
    if module_list is None:
        # Legacy: download all modules
        download_and_check_module_files()
    else:
        # Selected packages only
        download_and_check_module_files(module_list)

    if package_ids_include_optional_onnx(package_ids):
        download_optional_onnx_models()
        maybe_quantize_optional_onnx_models()

    if package_ids_include_pkuseg(package_ids):
        prepare_pkuseg()

    if shared.CACHE_UPDATED:
        shared.dump_cache()


def _as_list(v):
    if isinstance(v, list):
        return v
    return [v]


def _resolve_target_files(download_kwargs: Dict[str, Any]) -> tuple[list, list]:
    files = _as_list(download_kwargs.get('files', []))
    save_files = download_kwargs.get('save_files')
    if save_files is None:
        save_files = files
    save_files = _as_list(save_files)
    save_dir = download_kwargs.get('save_dir')
    if save_dir:
        save_files = [osp.join(save_dir, sf) for sf in save_files]
    sha = download_kwargs.get('sha256_pre_calculated')
    if isinstance(sha, list):
        sha_list = sha
    elif sha is None:
        sha_list = [None] * len(save_files)
    else:
        sha_list = [sha]
    if len(sha_list) < len(save_files):
        sha_list.extend([None] * (len(save_files) - len(sha_list)))
    return save_files, sha_list


def _all_files_ready(download_kwargs: Dict[str, Any]) -> bool:
    save_files, sha_list = _resolve_target_files(download_kwargs)
    if not save_files:
        return False
    for savep, sha in zip(save_files, sha_list):
        exists, valid, _ = check_local_file(savep, sha, cache_hash=True)
        if not exists or not valid:
            return False
    return True


def prepare_local_files_forall_with_summary() -> Dict[str, Any]:
    """
    Download model files and return a structured summary for UI/logging.
    """
    import utils.config as program_config
    from utils.model_packages import get_module_classes_for_packages, package_ids_include_pkuseg, package_ids_include_optional_onnx

    pcfg = program_config.pcfg
    package_ids = getattr(pcfg, "model_packages_enabled", None)
    effective_packages = list(package_ids) if package_ids is not None else ["legacy_all"]
    module_list = get_module_classes_for_packages(package_ids)
    if module_list is None:
        module_list = []
        for registered in [INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS]:
            for module_key in registered.module_dict.keys():
                module_list.append(registered.get(module_key))

    downloadable_modules = []
    for cls in module_list:
        if cls is None:
            continue
        if getattr(cls, 'download_file_on_load', False) or getattr(cls, 'download_file_list', None) is None:
            continue
        downloadable_modules.append(cls)

    LOGGER.info(
        "Model download start | package_ids=%s | module_count=%d | pkuseg=%s | optional_onnx=%s",
        effective_packages,
        len(downloadable_modules),
        package_ids_include_pkuseg(package_ids),
        package_ids_include_optional_onnx(package_ids),
    )

    downloaded = 0
    failed = 0
    skipped = 0
    failed_items: List[str] = []

    for cls in downloadable_modules:
        module_id = f"{cls.__module__}.{cls.__name__}"
        before_ready = True
        for download_kwargs in cls.download_file_list:
            if not _all_files_ready(download_kwargs):
                before_ready = False
                break
        module_success = True
        for download_kwargs in cls.download_file_list:
            ok = download_and_check_files(**download_kwargs)
            if not ok:
                module_success = False
        if module_success and before_ready:
            skipped += 1
        elif module_success:
            downloaded += 1
        else:
            failed += 1
            failed_items.append(module_id)
            LOGGER.error("Model download failed for module=%s", module_id)

    if package_ids_include_optional_onnx(package_ids):
        for idx, kw in enumerate(OPTIONAL_ONNX_MODELS):
            item_id = f"optional_onnx[{idx}]"
            before_ready = _all_files_ready(kw)
            ok = download_and_check_files(**kw)
            if ok and before_ready:
                skipped += 1
            elif ok:
                downloaded += 1
            else:
                failed += 1
                failed_items.append(item_id)
                LOGGER.error("Model download failed for optional ONNX asset=%s", item_id)
        maybe_quantize_optional_onnx_models()

    if package_ids_include_pkuseg(package_ids):
        pkuseg_before = True
        for kw in [
            {'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/postag.zip', 'files': ['features.pkl', 'weights.npz'], 'sha256_pre_calculated': ['17d734c186a0f6e76d15f4990e766a00eed5f72bea099575df23677435ee749d', '2bbd53b366be82a1becedb4d29f76296b36ad7560b6a8c85d54054900336d59a'], 'archived_files': 'postag.zip', 'save_dir': 'data/models/pkuseg/postag'},
            {'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/spacy_ontonotes.zip', 'files': ['features.msgpack', 'weights.npz'], 'sha256_pre_calculated': ['fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18', '5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e'], 'archived_files': 'spacy_ontonotes.zip', 'save_dir': 'data/models/pkuseg/spacy_ontonotes'},
        ]:
            if not _all_files_ready(kw):
                pkuseg_before = False
                break
        try:
            prepare_pkuseg()
            if pkuseg_before:
                skipped += 1
            else:
                downloaded += 1
        except Exception as e:
            failed += 1
            failed_items.append("pkuseg")
            LOGGER.error("Model download failed for pkuseg: %s", e)

    if shared.CACHE_UPDATED:
        shared.dump_cache()

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "package_ids": effective_packages,
        "module_count": len(downloadable_modules),
        "downloaded": downloaded,
        "failed": failed,
        "skipped": skipped,
        "failed_items": failed_items,
    }
    LOGGER.info(
        "Model download result | package_ids=%s | module_count=%d | downloaded=%d | failed=%d | skipped=%d | failed_items=%s",
        summary["package_ids"], summary["module_count"], summary["downloaded"], summary["failed"], summary["skipped"], summary["failed_items"]
    )
    return summary
