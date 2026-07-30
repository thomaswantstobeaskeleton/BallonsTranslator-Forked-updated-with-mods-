import json, os, traceback
import threading
import time
import os.path as osp
import copy
from typing import Callable, Optional

from . import shared
from .fontformat import FontFormat
from .structures import List, Dict, Config, field, nested_dataclass
from .logger import logger as LOGGER
from utils.credential_store import has_keyring, set_secret
from utils.secret_migration import migrate_secrets_to_keyring_if_possible, scrub_plaintext_secrets_for_save
from .io_utils import json_dump_nested_obj, np, serialize_np

class RunStatus:
    FIN_DET = 1
    FIN_OCR = 2
    FIN_INPAINT = 4
    FIN_TRANSLATE = 8
    FIN_ALL = 15


class OCRTextPostprocess:
    """Letter case applied to OCR results (upstream dmMaze/BallonsTranslator#1248).

    >>> OCRTextPostprocess.CAPITALIZE
    'capitalize'
    """

    NONE = 'none'
    CAPITALIZE = 'capitalize'
    UPPERCASE = 'uppercase'
    Valid = (NONE, CAPITALIZE, UPPERCASE)


@nested_dataclass
class ModuleConfig(Config):
    textdetector: str = 'ctd'
    ocr: str = "mit48px"
    inpainter: str = 'lama_large_512px'
    translator: str = "google"
    enable_detect: bool = True
    enable_dual_detect: bool = False
    textdetector_secondary: str = ''
    # When True, only add secondary (and tertiary) detector boxes that are outside primary bubbles (no in-bubble duplicates).
    secondary_detector_outside_bubble_only: bool = False
    enable_tertiary_detect: bool = False
    textdetector_tertiary: str = ''
    keep_exist_textlines: bool = False
    # Clip the inpaint mask to the detected text boxes (upstream dmMaze/BallonsTranslator#1213)
    filter_mask_by_bboxes: bool = False
    enable_ocr: bool = True
    enable_translate: bool = True
    enable_inpaint: bool = True
    # 是否在 OCR 后进行字体检测（默认不启用）
    ocr_font_detect: bool = False
    # Letter case applied to OCR results: none | capitalize | uppercase
    ocr_text_postprocess: str = OCRTextPostprocess.NONE
    textdetector_params: Dict = field(default_factory=lambda: dict())
    # Allow text detection to set box rotation when text is slanted (horizontal only). When off, horizontal boxes stay 0°.
    allow_detection_box_rotation: bool = False
    # Only apply rotation when angle is at least this many degrees from horizontal (e.g. 10 = ignore tiny slants).
    detection_rotation_threshold_degrees: float = 10.0
    ocr_params: Dict = field(default_factory=lambda: dict())
    translator_params: Dict = field(default_factory=lambda: dict())
    inpainter_params: Dict = field(default_factory=lambda: dict())
    translate_source: str = '日本語'
    translate_target: str = '简体中文'
    translate_by_textblock: bool = False
    check_need_inpaint: bool = True
    inpaint_tile_size: int = 0   # 0 = no tiling (recommended); set 512–1024 only if OOM
    inpaint_tile_overlap: int = 64   # overlap between tiles (px); only used when tile_size > 0
    # Optional: exclude text blocks by detector label from inpainting (e.g. leave scene text as-is). Off by default.
    inpaint_exclude_labels_d: bool = False
    inpaint_exclude_labels: str = ''  # comma-separated, case-insensitive (e.g. "other,scene")
    # When True, inpaint the whole image at once (no per-block crops). Uses more VRAM/slower but avoids per-block issues; try if Lama gives bad results.
    inpaint_full_image: bool = False
    # When True, wrap lama_mpe / lama_large_512px / AOT PyTorch models with torch.compile (CUDA only; first runs pay compile cost; variable crop shapes use dynamic=True).
    inpaint_torch_compile: bool = False
    # ONNX Runtime session options for lama_onnx / lama_manga_onnx (graph opts, memory; intra_op threads 0 = ORT default).
    inpaint_onnx_ort_graph_optimization_level: str = "all"  # all | extended | basic | disable
    inpaint_onnx_ort__mem_pattern: bool = True
    inpaint_onnx_ort__cpu_mem_arena: bool = True
    inpaint_onnx_ort_intra_op_num_threads: int = 0
    # Per-block inpainting: expand polygon mask vertically (descenders, halos). Critical for video subs + CTD; turn off if bubble edges look eaten.
    inpaint_block_mask_vertical_expand: bool = True
    load_model_on_demand: bool = False
    empty_runcache: bool = False
    # Optional: panel-aware reading order for block sorting (affects translation prompt order & typesetting sequence).
    _panel_order: bool = False
    # "auto" uses heuristic based on detected text orientation; "rtl" forces right-to-left; "ltr" forces left-to-right.
    panel_reading_direction: str = "auto"
    # Outside-speech-bubble (OSB) / text_free pipeline (optional, only works when detector sets blk.label, e.g. HF object det).
    _osb_pipeline: bool = False
    # Group nearby OSB boxes into larger regions (captions/SFX clusters).
    osb_group_nearby: bool = True
    osb_group_gap_px: int = 24
    # Drop OSB boxes that overlap bubble boxes by IoU (to avoid double-processing text inside bubbles).
    osb_exclude_bubble_iou: float = 0.10
    # After OCR, remove small margin page-number-like OSB blocks (e.g. "12"). Requires OCR d.
    osb_page_number_filter: bool = False
    osb_page_number_margin_ratio: float = 0.08
    # Probe OSB background and set readable fg/stroke defaults for rendering (does not override user styles later).
    osb_style_probe: bool = False
    # When True, OSB regions may be filled with median surrounding color if background is low-variance (faster/safer).
    osb_fast_fill: bool = False
    # Section 14: expand bubble boxes to fully contain overlapping OSB text so mask includes all text pixels.
    osb_expand_bubbles_with_osb: bool = True
    # Section 19: when OSB layout fails, retry with vertical stacking then restore original crop. Disable to only set restore_original_region on first failure.
    osb_layout_fallbacks_d: bool = True
    # Section 15: resolve overlapping mask regions by bisector split (and nudge for text boxes).
    resolve_mask_overlaps_bisector: bool = True
    # Section 16: cleaning quality — adaptive shrink at conjoined junctions, Otsu retry on failure.
    cleaning_adaptive_shrink_junction: bool = True
    cleaning_otsu_retry: bool = True
    # Section 17: colored/gradient bubbles — classify and inpaint text-only; re-sample brightness for contrast.
    colored_bubble_handling: bool = True
    colored_bubble_resample_brightness: bool = True
    # Translation workflow
    # - two_step: OCR -> text translator (default)
    # - one_step_vlm: OCR module performs translate-from-image and writes blk.translation directly (requires supported OCR, e.g. llm_ocr)
    translation_mode: str = "two_step"
    # Replace translation mode (manga-translator-ui style): input = raw image + pre-rendered translated image (same name in a folder).
    # Pipeline: detect+OCR on translated image -> text; detect on raw -> blocks; match by position; inpaint raw; render matched text.
    replace_translation_mode: bool = False
    replace_translation_translated_dir: str = ""  # Folder containing translated images (same filenames as project). Empty = off.
    # When True (default), on soft translation failure (e.g. timeout, parse error) use placeholder and continue batch. When False, show dialog and stop like critical errors.
    translation_soft_failure_continue: bool = True
    # Skip translation for pages that already have all blocks translated (non-empty translation). Speeds up re-runs.
    skip_already_translated: bool = False
    skip_satisfied_pipeline_steps: bool = False
    # Optional: merge nearby text blocks using collision-based grouping (Dango-style). Off by default;  for word-level OCR or many small blocks.
    merge_nearby_blocks_collision: bool = False
    merge_nearby_blocks_gap_ratio: float = 1.5  # vertical expansion ratio for horizontal merge; 1.5 = Dango-style
    # Only run collision merge when page has at least this many blocks (avoids merging normal bubble layouts; default 18).
    merge_nearby_blocks_min_blocks: int = 18
    # Translation caching (saves API costs for deterministic settings/reruns)
    translation_cache_d: bool = False
    translation_cache_deterministic_only: bool = True
    # In-session OCR cache: reuse OCR results for same image/model/language (comic-translate style). Reduces redundant OCR runs.
    ocr_cache_d: bool = True
    # When True, pipeline selects OCR by source language (e.g. Japanese → manga_ocr, Korean → preferred Korean OCR). Fallback: current OCR.
    ocr_auto_by_language: bool = False
    # Typesetting / layout (auto layout for translated text blocks)
    layout_optimal_breaks: bool = True
    layout_hyphenation: bool = True
    layout_collision_check: bool = True
    layout_collision_min_mask_ratio: float = 0.85
    layout_collision_max_retries: int = 3
    # Scoring/penalty weights for auto layout (non-CJK)
    layout_short_line_penalty: float = 80.0   # penalty per very short non-final line
    # Softer height overflow penalty so long English paragraphs can prefer fewer lines / better line quality
    # even if the block is slightly taller relative to the bubble.
    layout_height_overflow_penalty: float = 360.0  # lower = fewer lines, larger font; higher = slightly smaller font to fit height
    # Center text vertically (and horizontally as needed) inside each bubble/block (manga-translator-ui style).
    center_text_in_bubble: bool = False
    # Try larger font / fewer lines so text fits with fewer line breaks (test combinations; manga-translator-ui style).
    optimize_line_breaks: bool = False
    # When True, clamp text box size to the detected balloon and keep position fixed (no growing/moving after layout).
    layout_constrain_to_bubble: bool = True
    # After layout: check if box or text lines extend outside the bubble; shrink box or scale font to fix (no model).
    layout_check_overflow_after_layout: bool = True
    # Optional HF image-classification model to check if detection box is too large/small for bubble. Empty = fast geometric checks only. "builtin" = zero-shot CLIP.
    layout_box_size_check_model_id: str = ""
    # After auto layout, center each text box in its bubble (centroid). Skip boxes that are close to another (combined/overlapping bubbles).
    layout_center_in_bubble_after_autolayout: bool = True
    layout_center_in_bubble_min_gap_px: float = 40.0  # skip centering if another block is within this many pixels (edge-to-edge)
    # Preserve manual edits by default: bulk auto-layout passes skip textboxes marked user_adjusted.
    layout_skip_user_adjusted: bool = True
    # Layout judge: nudge text box toward bubble center and keep it away from bubble edges (no corners). Off = 0.
    layout_judge_d: bool = True
    layout_judge_margin_ratio: float = 0.06  # min margin from bubble edge (fraction of min(bubble_w, bubble_h)); e.g. 0.06 = 6%
    layout_judge_center_strength: float = 1.0  # 0 = no nudge, 1 = full nudge toward bubble center (right-click Judge and auto-layout)
    layout_judge_clamp_overflow: bool = True  # shrink/clamp box so it never extends outside the bubble
    # Optional small/fast model to assist judge (e.g. scale nudge by confidence). Empty = geometric only.
    layout_judge_model_id: str = "microsoft/resnet-18"  # Larger but still lightweight (~11.7M). Lighter: google/mobilenet_v2_1.0_224
    layout_judge_use_model: bool = False  # when True, run model on bubble crop and use score to modulate strength
    # Font scaling to fit bubble (2.1): min/max font size (pt) for auto layout; fit step clamps to this range.
    layout_font_size_min: float = 8.0
    layout_font_size_max: float = 72.0
    # User-facing automatic lettering profile: balanced | fit | readable.
    layout_auto_preset: str = "balanced"
    # When True, scale font so laid-out text fits inside bubble (ratio-based, then clamp to min/max). When False, only apply LAYOUT_* scale factors.
    layout_font_fit_bubble: bool = True
    # When True, use binary search to find the largest font size in [min,max] that fits the bubble (re-runs layout per trial; more accurate, slower).
    layout_font_binary_search: bool = True
    # Final automatic safety pass: after line breaking and box placement, shrink font only if rendered document still exceeds the textbox.
    layout_auto_final_fit_pass: bool = True
    # Make auto-lettering font fit more resolution-independent across small and huge pages.
    layout_scale_font_by_page_size: bool = True
    # Reference page area in megapixels used as neutral scale (1.0 = 1000x1000 page baseline).
    layout_scale_reference_megapixels: float = 1.0
    # Clamp auto page scale factor to avoid over/under-scaling on extreme resolutions.
    layout_scale_factor_min: float = 0.72
    layout_scale_factor_max: float = 1.58
    # Additional normalization using the current textbox area relative to page area,
    # so auto lettering stays consistent when bubbles are unusually tiny/huge.
    layout_scale_use_box_area: bool = True
    layout_scale_box_area_reference: float = 0.020
    # Production auto pass runs a final rendering-QA auto-fix stage for these issues.
    production_auto_pass_enable_qa_fixes: bool = True
    # Automatically choose a mask-safe inner text rectangle for curved/pointed bubbles instead of using the full contour bounding box.
    layout_use_mask_safe_area: bool = True
    # Balloon shape for Diamond-Text style layout: "auto" (detect from aspect ratio), "round", "elongated", "narrow", "diamond", "square", "bevel", "pentagon", "point". Affects insets and line-length scoring.
    layout_balloon_shape: str = "auto"
    # When "auto": which method(s) to use and in what order. Default is model-free contour/aspect-ratio detection; set a model ID to opt into model-based shape classification.
    layout_balloon_shape_auto_method: str = "contour_ratio"
    layout_balloon_shape_model_id: str = ""  # Optional. Heavier: prithivMLmods/Geometric-Shapes-Classification. Lighter: 0-ma/vit-geometric-shapes-tiny.
    # When "auto": comma-separated list of shape names the auto-detector is allowed to assign. Empty = all shapes. E.g. "round,elongated,square,bevel,point"
    layout_balloon_shape_auto_allowed: str = ""
    # Minimum line width (px) so short text (e.g. "pluck!") is not forced into 2–3 char lines; layout never uses a narrower width.
    layout_min_line_width_px: float = 80.0
    # Max line width as fraction of box width for free-standing text (no bubble). Lower = more lines, shorter lines. Only used when region_rect is None.
    layout_max_line_width_frac_no_bubble: float = 0.78
    # Extra penalty for 1-word lines in layout scorer (2.3); higher = engine strongly avoids single-word lines.
    layout_stub_penalty_1word: float = 2000.0
    # Rendering parity: when True, translation panel uses NoWrap so line breaks match the canvas bubble (no extra wrap).
    layout_panel_preserve_line_breaks: bool = False
        # Layout review agent
    layout_review_provider: str = "heuristic"  # heuristic | local_api | cloud_api
    layout_review_use_translator_settings: bool = True

    # Used when not inheriting from LLM_API_Translator, or as overrides.
    layout_review_api_provider: str = "OpenAI"
    layout_review_api_key: str = ""
    layout_review_api_endpoint: str = ""
    layout_review_override_model: str = ""

    layout_review_model_name: str = ""
    layout_review_prompt: str = (
        "Review manga/comic text box layout for overflow, off-center placement, "
        "bad readability, boxes outside bubbles, and text that should be resized or re-centered. "
        "Return only safe deterministic layout fix actions."
    )
    layout_review_temperature: float = 0.0
    layout_review_top_p: float = 1.0
    layout_review_max_tokens: int = 2048
    layout_review_include_page_screenshot: bool = True
    layout_review_screenshot_max_side: int = 1280
    layout_review_extra_params_json: str = "{}"
    finish_code: int = 15
    run_preset_name: str = 'Full'
    # --- Section 6 / 6.1: Image upscaling & per-stage resizing ---
    # Global default min side for OCR crop upscale (0 = off). Per-OCR params (e.g. EasyOCR upscale_min_side) override when set.
    ocr_upscale_min_side: int = 0
    # Initial upscale: before detection/OCR (improves small text). Pipeline runs on upscaled image then downscales results to original size.
    image_upscale_initial: bool = False
    image_upscale_initial_factor: float = 2.0
    # Final output upscale: 2x (or factor) when saving result image.
    image_upscale_final: bool = False
    image_upscale_final_factor: float = 2.0
    # Auto-scale pipeline params by image area (processing_scale = sqrt(area/1e6)) for fonts, padding, morphology, thresholds.
    processing_scale_enabled: bool = True
    # Per-stage resize policy: none | lanczos (model/model_lite reserved for future).
    upscale_policy_initial: str = "lanczos"
    upscale_policy_final: str = "lanczos"
    # Optional: lightweight colorization of grayscale pages when saving final result.
    enable_colorization: bool = False
    colorization_strength: float = 0.6  # 0–1; blend between grayscale and colorized
    # Colorization backend: 'simple' (soft twilight), 'manga_vibrant', 'cool', etc.
    colorization_backend: str = "simple"
    # Section 7: Caching + memory / stability
    pipeline_cache_enabled: bool = False  # When True, in-memory pipeline cache can be used (get_pipeline_cache(True))
    inpaint_spill_to_disk_after_blocks: int = 0  # When >0, write intermediate inpainted image to temp file every N blocks to reduce peak RAM/VRAM (e.g. 8 or 12)
    # --- Video translator (Pipeline → Video translator...) ---
    video_translator_usage_preset: str = "balanced"  # Preset name applied from Usage preset dropdown: custom | dont_miss_text | balanced | max_speed | anime | documentary.
    video_translator_sample_every_frames: int = 30  # Process every N frames; in-between frames reuse last result (reduces flicker, faster).
    video_translator_enable_detect: bool = True
    video_translator_enable_ocr: bool = True
    video_translator_enable_translate: bool = True
    video_translator_enable_inpaint: bool = True
    video_translator_last_input_path: str = ''   # Last used input path (persisted)
    video_translator_last_output_path: str = '' # Last used output path (persisted)
    video_translator_output_codec: str = 'mp4v' # OpenCV fourcc: mp4v, avc1, XVID, etc. Empty = try mp4v then avc1.
    video_translator_region_preset: str = 'full'  # full | bottom_15 | bottom_20 | bottom_25 | bottom_30 — only process blocks in that region (faster, fewer false positives).
    video_translator_use_scene_detection: bool = False  # Run pipeline only on scene-change frames (reuse result until next scene); saves work.
    video_translator_scene_threshold: float = 30.0  # Histogram diff threshold for scene change (higher = fewer scene cuts).
    video_translator_temporal_smoothing: bool = False  # Blend current result with previous in subtitle region to reduce flicker.
    # Translation Assist (Phase 5 baseline)
    enable_translation_assist: bool = True
    auto_query_tm: bool = True
    auto_query_glossary: bool = True
    auto_query_concordance: bool = False
    auto_query_sfx: bool = True
    auto_run_mt_candidates: bool = False
    max_mt_candidates: int = 6
    preferred_assist_providers: List = field(default_factory=lambda: ["current_translator"])
    default_assist_prompt_profile: str = "dialogue"
    video_translator_temporal_alpha: float = 0.25  # Weight of previous frame in blend (0=no smoothing, 0.5=half previous).
    video_translator_use_ffmpeg: bool = False  # Encode output with FFmpeg (libx264) for better compatibility than OpenCV.
    video_translator_ffmpeg_path: str = ''  # Path to ffmpeg.exe if not on PATH (e.g. C:\ffmpeg\bin\ffmpeg.exe).
    video_translator_ffmpeg_crf: int = 18  # FFmpeg CRF (0–51, lower=better quality); 18 = higher quality than 23, avoids very low bitrate.
    video_translator_ffmpeg_preset: str = "medium"  # libx264 preset: ultrafast, superfast, veryfast, faster, fast, medium, slow; faster = encode speed.
    video_translator_ffmpeg_hw_encoder: str = "none"  # GPU encoder: none | nvenc | qsv | auto (NVIDIA NVENC or Intel QSV when available).
    video_translator_video_bitrate_kbps: int = 0  # Target bitrate in kbps when using FFmpeg; 0 = use CRF only. Set e.g. 9600 to match source.
    video_translator_skip_detect: bool = False  # Use fixed subtitle region only (no detector); region_preset defines the band.
    video_translator_detect_no_inpaint: bool = False  # Run detection/OCR/translation but skip inpainting (burn subtitles directly).
    # When skip_detect is on, use native bottom-band OCR/inpaint semantics:
    # keep OCR/inpaint focused on the whole subtitle band instead of skipping inpaint for the fixed block.
    video_translator_bottom_band_native_mode: bool = False
    video_translator_prefetch_frames: int = 2  # 0 = off; 2–3 = decode next frames in a separate thread to avoid I/O blocking (OCR path only).
    video_translator_background_writer: bool = True  # OCR path always uses a separate encode thread; option kept for compatibility.
    video_translator_use_two_pass_ocr_burn_in: bool = True  # OCR burn-in path: pass 1 detect/OCR/inpaint + async translate, pass 2 burn timed cues.
    video_translator_two_stage_keyframes: bool = False  # Skip full pipeline when subtitle region content hash unchanged (OCR path only).
    video_translator_two_stage_force_refresh_every_frames: int = 0  # 0 = off; otherwise force a full pipeline run at least every N frames.
    video_translator_two_stage_new_line_diff_threshold: float = 8.0  # ROI band mean abs-diff threshold to detect "new line" changes.
    video_translator_auto_catch_subtitle_on_skipped_frames: bool = True  # On skipped frames, auto-run pipeline when subtitle band changes enough (helps first-appearance capture).
    video_translator_auto_catch_diff_threshold: float = 0.0  # 0 = auto from sample_every_frames/fps; else fixed pixel-diff threshold.
    video_translator_adaptive_detector_roi: bool = False  # Crop detector to ROI around last subtitle blocks (when available) to speed up detection.
    video_translator_adaptive_detector_roi_padding_frac: float = 0.15  # ROI padding fraction around last subtitle union bbox.
    video_translator_adaptive_detector_roi_start_seconds: float = 0.0  # Delay adaptive ROI activation until this timestamp (sec) to avoid early-frame bias.
    video_translator_overlap_inpaint: bool = False  # Overlap inpainting with OCR/translation within a single pipeline run (may increase GPU contention).
    video_translator_overlap_inpaint_require_cpu: bool = True  # If true, only overlap when inpainter is on CPU.
    # Video frames from OpenCV are BGR; LaMa and most neural inpainters expect RGB. Turning off restores legacy behavior.
    video_translator_inpaint_bgr_to_rgb: bool = True
    # Extra vertical dilation + slight downward shift on the inpaint mask so descenders, outlines, and soft edges are included.
    video_translator_inpaint_subtitle_mask_expand: bool = True
    video_translator_ocr_cache_geo_quantization: int = 8  # Quantization grid (px) for OCR cache geometry keys; higher = more hits but more reuse risk.
    # Temporal OCR stabilization (video OCR): vote over recent frames to reduce per-frame character jitter.
    video_translator_ocr_temporal_stability: bool = True
    video_translator_ocr_temporal_window: int = 5  # Number of recent frames considered for OCR text vote.
    video_translator_ocr_temporal_min_votes: int = 2  # Minimum agreeing frames required to replace current OCR text.
    video_translator_ocr_temporal_geo_quantization: int = 24  # Region quantization (px) for matching the same subtitle region across frames.
    video_translator_flow_fixer_cache_size: int = 200  # Max flow-fixer memoization entries per run (prev+new translation pairs).
    video_translator_export_srt: bool = False  # Write an SRT file alongside the output video with timed subtitles.
    # When True (recommended), burn-in uses timed cue timeline rendering per frame
    # instead of relying on cached text-region composites. More deterministic, supports
    # stacked overlaps, and avoids cached-text priority flicker.
    video_translator_prefer_timed_burn_in: bool = True
    video_translator_subtitle_style: str = "default"  # Burn-in style: default | anime | documentary (VideoCaptioner-inspired).
    video_translator_subtitle_font: str = ""  # Optional path to .ttf/.otf for burn-in subtitles; empty = Arial/DejaVu/fallback.
    video_translator_soft_subs_only: bool = False  # When True, output original video + SRT only (no burn-in); fast, player-controlled subs.
    video_translator_inpaint_only_soft_subs: bool = False  # When True, run pipeline (inpaint) but do not burn-in; output inpainted video + SRT/ASS/VTT (no double subs).
    video_translator_mux_srt_into_video: bool = False  # When inpaint_only_soft_subs, mux SRT as subtitle stream into the output video file.
    video_translator_source: str = "ocr"  # ocr = hardcoded subtitles (detect+OCR); asr = audio speech-to-text.
    video_translator_asr_model: str = "base"  # faster-whisper model: tiny, base, small, medium, large-v2, large-v3.
    video_translator_asr_device: str = "cuda"  # cuda or cpu for ASR.
    video_translator_asr_language: str = ""  # Empty = auto-detect; e.g. ja, en, zh.
    video_translator_asr_vad_filter: bool = True  # VAD filter to reduce ASR hallucinations (VideoCaptioner-style).
    video_translator_asr_chunk_seconds: float = 2400.0  # When >0 and duration >= threshold, transcribe in time chunks (seconds per slice).
    video_translator_asr_long_audio_threshold_seconds: float = 5400.0  # Minimum duration (sec) to enable chunked ASR; 0 = never chunk by duration.
    video_translator_asr_checkpoint_resume: bool = True  # Save/resume chunk progress to temp JSON when chunking is active.
    video_translator_export_ass: bool = False  # Export ASS subtitle file alongside video.
    video_translator_export_vtt: bool = False  # Export WebVTT subtitle file alongside video.
    video_translator_glossary: str = ""  # Optional glossary/script hint for LLM (OCR/ASR correction and translation).
    video_translator_lock_watermark_lines: bool = False  # If true, lines matching regex are not translated (kept as source text).
    video_translator_lock_watermark_regex: str = r"备案号[:：]?\s*\d{8,}"  # Regex for watermark-like lines to lock/ignore in translation.
    # VideoCaptioner-style subtitle NLP throughput (LLM_API only): split one translate batch into multiple API calls
    # and optionally run them in parallel. Does not add extra vision/OCR threads — translation stage only.
    # 0 = one API request per translate_textblk_lst (all cues in one JSON batch; can be hundreds of lines).
    # 20–40 is a practical range for ASR / existing-subs so prompts stay small; also used for subtitle-file translate.
    video_translator_nlp_chunk_size: int = 32
    video_translator_nlp_max_workers: int = 1  # Parallel LLM requests when chunking produces multiple chunks (2–4 typical). Watch API rate limits.
    # When a batched LLM reply has wrong id/count after retries, re-translate each cue one-by-one (video_* only).
    video_translator_llm_strict_alignment_fallback: bool = True
    # Fix empty lines and obvious source-echo (CJK/JP/KR) with a few per-cue calls before post-check per-line retries.
    video_translator_llm_per_line_quality_fix: bool = True
    # Re-translate cues whose English largely repeats the previous cue (split-ASR / batch “flow” echo).
    video_translator_llm_redundant_continuation_fix: bool = True
    video_translator_qwen35_allow_aux_passes: bool = False  # Allow Qwen3.5-4B LM Studio OCR-correction/reflection passes (can be slower/less stable JSON).
    video_translator_series_context_path: str = ""  # Optional series context path (folder or ID) for glossary/context; same as project series context.
    # Koharu-inspired LLM QA/consistency controls
    enable_glossary_enforcement: bool = True
    llm_glossary_map: Dict = field(default_factory=dict)
    enable_back_translation_qa: bool = False
    back_translation_drift_threshold: float = 0.58
    llm_token_budget: int = 420
    enable_text_normalization: bool = False
    text_normalization_profile: str = "balanced"
    runtime_http_timeout_sec: float = 60.0
    runtime_http_retries: int = 1
    data_path_override: str = ""
    automation_api_enabled: bool = False
    automation_api_port: int = 39542
    automation_api_key: str = ""
    credential_use_plaintext_fallback: bool = False
    credential_migration_done: bool = False
    automation_api_job_history_limit: int = 200
    automation_api_job_log_limit: int = 200
    user_replace_profiles: Dict = field(default_factory=dict)
    vertical_cjk_rotate_latin: bool = True
    vertical_cjk_punctuation_hang: bool = True
    pipeline_retry_detect: int = 1
    pipeline_retry_ocr: int = 1
    pipeline_retry_translate: int = 1
    pipeline_retry_inpaint: int = 1
    video_translator_asr_sentence_break: bool = False  # LLM merges/splits ASR segments into natural sentences.
    video_translator_sentence_merge_by_punctuation: bool = True  # Rule-based: merge adjacent short cues until sentence punctuation before translation.
    video_translator_sentence_merge_max_seconds: float = 8.0  # Upper bound for one merged cue duration to avoid over-long subtitle chunks.
    video_translator_asr_audio_separation: bool = False  # Separate vocals before ASR (demucs; reduces music noise).
    video_translator_asr_guided_detect_inpaint: bool = False  # ASR-guided detect/inpaint: detect at subtitle segment boundaries, reuse boxes during active segment.
    video_translator_asr_guided_midpoint_refresh: bool = True  # In ASR-guided mode, refresh detection once near segment midpoint.
    video_translator_last_batch_output_dir: str = ""  # Last used output directory for batch video processing.
    # Flow fixer: local model (Ollama/LM Studio) to improve subtitle flow without extra cloud API calls.
    video_translator_flow_fixer_enabled: bool = False
    video_translator_use_flow_fixer_for_corrections: bool = False  # Use flow fixer model for OCR correction, ASR correction, and reflection (same model as flow fixer).
    video_translator_flow_fixer: str = "none"  # none | local_server | openrouter | openai
    video_translator_flow_fixer_context_lines: int = 20  # Max previous subtitle lines sent to flow fixer (1–50). Revisions apply this far back.
    video_translator_flow_fixer_strict_single_line_review: bool = False  # If true, do not skip flow-fixer API on single-line/no-context updates.
    video_translator_flow_fixer_server_url: str = "http://localhost:1234/v1"  # LM Studio default; Ollama: http://localhost:11434/v1
    video_translator_flow_fixer_model: str = ""  # Model name for local_server (required: type exact name for LM Studio/Ollama)
    video_translator_flow_fixer_max_tokens: int = 512
    video_translator_flow_fixer_timeout: float = 30.0
    video_translator_flow_fixer_enable_reasoning: bool = True  # Flow fixer / timeline review: reasoning when supported (JSON still extracted after think).
    video_translator_flow_fixer_reasoning_effort: str = "medium"  # low | medium | high
    video_translator_post_review_enabled: bool = True  # After run/cancel, run one global subtitle flow review before final sidecar output.
    video_translator_post_review_apply_on_cancel: bool = True  # Also run post review when run is cancelled (partial timeline).
    # When no dedicated flow-fixer model is set (or it is no-op), use the main translator for the same flow JSON pass.
    video_translator_post_review_use_main_translator: bool = True
    video_translator_post_review_main_llm_max_tokens: int = 8192  # Flow-fix chunks need room for JSON (reasoning models may use prose first).
    video_translator_post_review_enable_reasoning: bool = True  # Main-LLM post-review: use reasoning (batched translation forces reasoning off separately).
    video_translator_post_review_reasoning_effort: str = "medium"  # low | medium | high for post-review when reasoning is on.
    video_translator_post_review_chunk_size: int = 80  # Timeline lines per post-review call.
    video_translator_post_review_context_lines: int = 20  # Previous lines provided as context for each post-review chunk.
    video_translator_ab_model_a: str = ""  # Optional saved model A for subtitle A/B compare tool.
    video_translator_ab_model_b: str = ""  # Optional saved model B for subtitle A/B compare tool.
    video_translator_ab_sample_size: int = 200  # Sample size for subtitle A/B compare tool.
    video_translator_ab_custom_lines: str = ""  # Newline-separated custom lines for subtitle A/B compare presets.
    # OpenRouter flow fixer (when flow_fixer == "openrouter"): second model for flow only
    video_translator_flow_fixer_openrouter_apikey: str = ""
    video_translator_flow_fixer_openrouter_model: str = "google/gemma-3n-e2b-it:free"
    # OpenAI / ChatGPT flow fixer (when flow_fixer == "openai"): use ChatGPT credits
    video_translator_flow_fixer_openai_apikey: str = ""
    video_translator_flow_fixer_openai_model: str = "gpt-4o-mini"

    def get_params(self, module_key: str, for_saving=False) -> dict:
        d = self[module_key + '_params']
        if not for_saving:
            return d
        sd = {}
        for module_key, module_params in d.items():
            if module_params is None:
                continue
            saving_module_params = {}
            sd[module_key] = saving_module_params
            for pk, pv in module_params.items():
                if pk in {'description'}:
                    continue
                if pk.startswith('__'):
                    continue
                if isinstance(pv, dict):
                    pv = pv['value']
                saving_module_params[pk] = pv
        return sd

    def get_saving_params(self, to_dict=True):
        params = copy.copy(self)
        params.ocr_params = self.get_params('ocr', for_saving=True)
        params.inpainter_params = self.get_params('inpainter', for_saving=True)
        params.textdetector_params = self.get_params('textdetector', for_saving=True)
        params.translator_params = self.get_params('translator', for_saving=True)
        if to_dict:
            return params.__dict__
        return params
    
    def stage_enabled(self, idx: int):
        if idx == 0:
            return self.enable_detect
        elif idx == 1:
            return self.enable_ocr
        elif idx == 2:
            return self.enable_translate
        elif idx == 3:
            return self.enable_inpaint
        else:
            raise Exception(f'not supported stage idx: {idx}')
        
    def all_stages_disabled(self):
        return (self.enable_detect or self.enable_ocr or self.enable_translate or self.enable_inpaint) is False

    def __post_init__(self):
        if self.ocr_text_postprocess not in OCRTextPostprocess.Valid:
            self.ocr_text_postprocess = OCRTextPostprocess.NONE
        self.update_finish_code()

    def update_finish_code(self):
        self.finish_code = self.enable_detect * RunStatus.FIN_DET + \
            self.enable_ocr * RunStatus.FIN_OCR + \
                self.enable_translate * RunStatus.FIN_TRANSLATE + \
                    self.enable_inpaint * RunStatus.FIN_INPAINT
        

@nested_dataclass
class DrawPanelConfig(Config):
    pentool_color: List = field(default_factory=lambda: [0, 0, 0, 255])  # [r, g, b, a]
    pentool_width: float = 30.
    pentool_shape: int = 0
    inpainter_width: float = 30.
    inpainter_shape: int = 0
    inpaint_hardness: int = 100  # 100 = hard edge, 0 = soft/feathered
    current_tool: int = 0
    rectool_auto: bool = False
    rectool_method: int = 0
    rectool_shape: int = 0  # 0 = Rectangle, 1 = Ellipse (#35)
    recttool_dilate_ksize: int = 0
    recttool_erode_ksize: int = 0
    # Optional: SAM2/SAM3 refinement for balloon masks (used by the inpaint mask-seg method).
    # Requires transformers with SAM2/SAM3 support; if unavailable, the app falls back gracefully.
    sam_maskrefine_model_id: str = "facebook/sam2.1-hiera-large"
    # Empty => auto-select ("cuda" if available else "cpu"). You can also set "cuda" / "cpu".
    sam_maskrefine_device: str = ""
    # Expand the prompt box around the coarse mask by this many pixels (crop-local coords).
    sam_maskrefine_padding_px: int = 12

@nested_dataclass
class ProgramConfig(Config):

    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    drawpanel: DrawPanelConfig = field(default_factory=lambda: DrawPanelConfig())
    global_fontformat: FontFormat = field(default_factory=lambda: FontFormat())
    recent_proj_list: List = field(default_factory=lambda: list())
    show_page_list: bool = False
    imgtrans_paintmode: bool = False
    imgtrans_textedit: bool = True
    imgtrans_textblock: bool = True
    mask_transparency: float = 0.
    original_transparency: float = 0.
    open_recent_on_startup: bool = True
    recent_proj_list_max: int = 14
    # When True, show the welcome screen on startup when no project is opened (manhua-translator / Komakun style).
    show_welcome_screen: bool = True
    # When True, check for and pull GitHub updates on startup. Can cause issues or bad results; use with caution.
    auto_update_from_github: bool = False
    show_model_download_result_dialog: bool = True
    show_startup_health_dialog: bool = True
    logical_dpi: int = 0
    confirm_before_run: bool = True
    let_fntsize_flag: int = 0
    let_fntstroke_flag: int = 0
    let_fntcolor_flag: int = 0
    let_fnt_scolor_flag: int = 0
    let_fnteffect_flag: int = 1
    let_alignment_flag: int = 0
    let_writing_mode_flag: int = 0
    let_family_flag: int = 0
    let_autolayout_flag: bool = True
    let_uppercase_flag: bool = True
    let_show_only_custom_fonts_flag: bool = False
    let_textstyle_indep_flag: bool = False
    text_styles_path: str = osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')
    default_text_style_name: str = ''  # Preset name to show as selected (blue) on startup when using Save as default
    fsearch_case: bool = False
    fsearch_whole_word: bool = False
    fsearch_regex: bool = False
    fsearch_range: int = 0
    gsearch_case: bool = False
    gsearch_whole_word: bool = False
    gsearch_regex: bool = False
    gsearch_range: int = 0
    darkmode: bool = True
    bubbly_ui: bool = True
    accent_color_hex: str = ''  # Theme customizer: e.g. #1E93E5 (blue) or #9B59B6 (purple). Empty = use theme default.
    app_font_family: str = ''   # Theme customizer: app-wide font. Empty = system default.
    app_font_size: int = 0      # Theme customizer: app-wide font size. 0 = system default.
    use_custom_cursor: bool = False
    custom_cursor_path: str = ''
    textselect_mini_menu: bool = True
    fold_textarea: bool = False
    show_source_text: bool = True
    show_trans_text: bool = True
    saladict_shortcut: str = "Alt+S"
    search_url: str = "https://www.google.com/search?q="
    ocr_sublist: List = field(default_factory=lambda: list())
    # When True, after OCR any block whose recognized text is empty is REMOVED from the page (mask/inpaint restored).
    # Turn OFF to keep all text boxes even when OCR returns nothing (avoids "deleting" boxes when OCR fails or returns 1 char).
    restore_ocr_empty: bool = False
    pre_mt_sublist: List = field(default_factory=lambda: list())
    mt_sublist: List = field(default_factory=lambda: list())
    display_lang: str = field(default_factory=lambda: shared.DEFAULT_DISPLAY_LANG) # to always apply shared.DEFAULT_DISPLAY_LANG
    imgsave_quality: int = 100
    imgsave_webp_lossless: bool = False
    imgsave_ext: str = '.png'
    intermediate_imgsave_ext: str = '.png'
    supersampling_factor: int = 1  # 1 = off, 2..4 render at Nx then downscale for smoother edges
    # Section 10: Canvas view mode for QA (original / debug boxes-masks / translated / normal).
    canvas_view_mode: str = "normal"  # "normal" | "original" | "debug" | "translated"
    show_text_style_preset: bool = True
    expand_tstyle_panel: bool = True
    show_text_effect_panel: bool = True
    expand_teffect_panel: bool = True
    text_advanced_format_panel: bool = True
    expand_tadvanced_panel: bool = True
    config_panel_font_scale: float = 1.0
    # When False, the Config panel hides expert/advanced layout-engine knobs to reduce visual clutter for new users.
    show_advanced_settings: bool = False
    default_device: str = ''
    unload_after_idle_minutes: int = 0
    ocr_spell_check: bool = False
    # Underline misspelled words in the source / translation editors (issue #12).
    spell_check_highlight: bool = True
    manga_source_lang: str = 'en'
    manga_source_data_saver: bool = False
    manga_source_download_dir: str = ''
    manga_source_request_delay: float = 0.3
    manga_source_open_after_download: bool = False
    manga_source_playwright_headless: bool = True
    manga_source_translate_raw_search: bool = True  # For raw sources: translate search query to Japanese/Korean/Chinese
    manga_source_show_experimental: bool = True
    manga_source_base_url_overrides: Dict[str, str] = field(default_factory=dict)
    manga_source_filter: str = "all"
    # Model packages to download at startup (None = legacy "all"; ["core"] = minimal). See utils.model_packages.
    model_packages_enabled: Optional[List[str]] = field(default_factory=lambda: ["core"])
    # True when first-run explicitly chose "Skip all downloads, local-only modules".
    offline_local_only_mode: bool = False
    # When True, module dropdown tooltips show tier badges (Stable/Beta/Experimental/External-heavy).
    show_module_tier_badges_in_tooltips: bool = True
    # Last startup/retry model download status for support/debug (timestamp, package ids, and result counts).
    model_download_last_status: Dict = field(default_factory=dict)
    # User-facing preset IDs selected on first run (or ["custom"] for manual package selection).
    model_package_preset_ids: List[str] = field(default_factory=lambda: ["balanced_default"])
    # When True, show all modules in detector/OCR/translator dropdowns (including not downloaded or incompatible). When False, only show ready modules.
    dev_mode: bool = False
    # Temporary: when enabled, emit structured diagnostic logs for UI actions and pipeline stage transitions.
    diagnostic_mode: bool = False
    shortcuts: Dict = field(default_factory=dict)
    auto_region_merge_after_run: str = 'never'  # 'never' | 'all_pages' | 'current_page'
    region_merge_settings: Dict = field(default_factory=dict)  # Region merge tool dialog (persisted)
    context_menu: Dict = field(default_factory=dict)  # Canvas right-click: action key -> visible (default True)
    context_menu_pinned: List = field(default_factory=list)  # Action keys to show at top of right-click menu (order preserved)
    context_run_macros: List = field(default_factory=lambda: [
        {'id': 'detect_ocr_translate', 'label': 'Macro: Detect+OCR+Translate', 'mode': 1, 'detect_first': True},
        {'id': 'ocr_translate_inpaint', 'label': 'Macro: OCR+Translate+Inpaint', 'mode': 2, 'detect_first': False},
    ])
    huggingface_token: str = ''  # Optional: gated models + faster HF downloads (Xet). Prefer env HF_TOKEN to avoid storing in config.
    translator_last_model_by_provider: Dict = field(default_factory=dict)  # Section 9: last-used model per LLM provider
    # When True, the "Add Open in BallonsTranslator to context menu?" dialog has been shown once (first launch); don't show again.
    windows_context_menu_offered: bool = False
    # Release detector/OCR/inpainter/translator caches and gc after pipeline finishes (all pages). Reduces RSS when idle.
    release_caches_after_batch: bool = False
    # Manual mode: run pipeline on current page only (comic-translate style). Run button still works but processes one page.
    manual_mode: bool = False
    # When True (default), full run and batch skip pages marked as "ignored" in the page list.
    skip_ignored_in_run: bool = True
    # Automatically mark pages as translated when a pipeline run satisfies the enabled stage finish contract.
    auto_mark_translated_pages: bool = True
    # Rendering/text-formatting defaults and diagnostics.
    render_default_writing_mode: str = "auto"
    render_default_fit_mode: str = "shrink"
    render_default_line_break_strategy: str = "auto"
    render_default_reading_order: str = "auto"
    render_overflow_warnings: bool = True
    render_diagnostics_overlay: bool = False
    render_auto_polish_on_ocr: bool = True
    render_default_font_family: str = ""
    render_default_stroke_width: float = 0.08
    render_default_secondary_stroke_width: float = 0.0
    render_default_secondary_stroke_color: List = field(default_factory=lambda: [255, 255, 255])
    render_default_shadow_radius: float = 0.0
    render_default_shadow_strength: float = 1.0
    render_default_text_padding: float = 2.0
    render_atomic_fit_target_fill: float = 0.78
    render_atomic_fit_max_expand: float = 1.22
    render_atomic_fit_profile: str = "balanced"
    render_fallback_fonts_latin: str = "Arial, Noto Sans, DejaVu Sans"
    render_fallback_fonts_cjk: str = "Noto Sans CJK JP, Noto Sans CJK SC, Yu Gothic, Microsoft YaHei"
    render_fallback_fonts_korean: str = "Noto Sans CJK KR, Malgun Gothic"
    render_fallback_fonts_rtl: str = "Noto Naskh Arabic, Arial, Segoe UI"
    render_fallback_fonts_emoji: str = "Noto Color Emoji, Segoe UI Emoji, Apple Color Emoji"
    render_favorite_fonts: str = ""  # Comma-separated favorites shown first in text formatting font picker.
    render_recent_fonts: str = ""  # Recently applied text-panel font families for quick reuse.
    render_custom_manga_presets: Dict = field(default_factory=dict)
    export_open_folder_after_batch: bool = False
    export_include_unrendered_pages: bool = False
    export_filename_template: str = "{index:03d}"
    text_editor_top_padding: int = 14
    # Smooth scroll: animate scroll position on wheel (ms). 0 = off. 80–200 = subtle enhanced feel.
    smooth_scroll_duration_ms: int = 0
    # When True, briefly apply motion blur to scroll area viewport during scroll (can be costly).
    motion_blur_on_scroll: bool = False
    # When True, shorten or disable UI animations (accessibility / preference). Durations → 0 or minimal; scale and motion-blur effects off.
    reduce_motion: bool = False

    @staticmethod
    def default_downloaded_chapters_dir() -> str:
        """Return the default folder for downloaded chapters; creates it if it does not exist."""
        path = osp.join(osp.expanduser("~"), "BallonsTranslator", "Downloaded Chapters")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            LOGGER.warning("Could not create default download folder %s: %s", path, e)
        return path

    @staticmethod
    def load(cfg_path: str):
        
        with open(cfg_path, 'r', encoding='utf8') as f:
            config_dict = json.loads(f.read())

        # for backward compatibility
        if 'dl' in config_dict:
            dl = config_dict.pop('dl')
            if not 'module' in config_dict:
                if 'textdetector_setup_params' in dl:
                    textdetector_params = dl.pop('textdetector_setup_params')
                    dl['textdetector_params'] = textdetector_params
                if 'inpainter_setup_params' in dl:
                    inpainter_params = dl.pop('inpainter_setup_params')
                    dl['inpainter_params'] = inpainter_params
                if 'ocr_setup_params' in dl:
                    ocr_params = dl.pop('ocr_setup_params')
                    dl['ocr_params'] = ocr_params
                if 'translator_setup_params' in dl:
                    translator_params = dl.pop('translator_setup_params')
                    dl['translator_params'] = translator_params
                config_dict['module'] = dl

        if 'module' in config_dict:
            module_cfg = config_dict['module']
            trans_params = module_cfg['translator_params']
            repl_pairs = {'baidu': 'Baidu', 'caiyun': 'Caiyun', 'chatgpt': 'ChatGPT', 'Deepl': 'DeepL', 'papago': 'Papago'}
            for k, i in repl_pairs.items():
                if k in trans_params:
                    trans_params[i] = trans_params.pop(k)
            if module_cfg['translator'] in repl_pairs:
                module_cfg['translator'] = repl_pairs[module_cfg['translator']]

        # Legacy: configs that lack this key or have null used to download all models. Default to core-only (Issue #15).
        if config_dict.get("model_packages_enabled") is None:
            config_dict["model_packages_enabled"] = ["core"]
        if "offline_local_only_mode" not in config_dict:
            config_dict["offline_local_only_mode"] = False
        if not config_dict.get("model_package_preset_ids"):
            config_dict["model_package_preset_ids"] = ["balanced_default"]

        return ProgramConfig(**config_dict)
    

pcfg = ProgramConfig()
text_styles: List[FontFormat] = []
active_format: FontFormat = None

# Default keys for canvas context menu visibility (all True = show).
CONTEXT_MENU_DEFAULT = {
    'edit_copy': True, 'edit_paste': True, 'edit_copy_trans': True, 'edit_paste_trans': True,
    'edit_copy_src': True, 'edit_paste_src': True, 'edit_delete': True, 'edit_delete_recover': True,
    'edit_clear_src': True, 'edit_clear_trans': True, 'edit_select_all': True,
    'text_spell_src': True, 'text_spell_trans': True, 'text_trim': True, 'text_upper': True, 'text_lower': True,
    'text_strikethrough': True, 'text_gradient': True, 'text_on_path': True,
    'block_merge': True, 'block_split': True, 'block_move_up': True, 'block_move_down': True,
    'block_triage_add': True, 'block_triage_mark_reviewed': True,
    'create_textbox': True,
    'overlay_import': True, 'overlay_clear': True,
    'transform_free': True, 'transform_reset_warp': True, 'transform_warp_preset': True,
    'order_bring_front': True, 'order_send_back': True,
    'format_apply': True, 'format_layout': True, 'format_auto_fit': True, 'format_fit_to_bubble': True, 'format_smart_auto_fit': True, 'format_atomic_bubble_fit': True, 'format_polish_typography': True, 'format_auto_fit_binary': True, 'format_re_auto_fit_selected': True, 'format_re_auto_fit_page': True, 'format_re_auto_fit_all': True, 'format_balloon_shape': True, 'format_resize_to_fit_content': True, 'format_fit_to_mask_safe_box': True, 'format_center_in_bubble': True, 'format_angle': True, 'format_squeeze': True,
    'format_layout_review_selected': True, 'format_layout_review_page': True, 'format_layout_review_config': True,
    'run_detect_region': True, 'run_detect_page': True, 'run_translate': True, 'run_ocr': True,
    'run_ocr_translate': True, 'run_ocr_translate_inpaint': True, 'run_inpaint': True,
    'review_ocr_triage_page': True, 'review_translation_qa_page': True, 'review_auto_extract_glossary_page': True,
    'download_image': True,
}

# Section 9: canonical key order when saving config (clean diffs, easier debugging)
CONFIG_KEY_ORDER = (
    "module", "drawpanel", "global_fontformat", "recent_proj_list", "show_page_list",
    "imgtrans_paintmode", "imgtrans_textedit", "imgtrans_textblock", "mask_transparency", "original_transparency",
    "open_recent_on_startup", "recent_proj_list_max", "show_welcome_screen", "auto_update_from_github", "show_model_download_result_dialog", "show_startup_health_dialog", "logical_dpi", "confirm_before_run",
    "let_fntsize_flag", "let_fntstroke_flag", "let_fntcolor_flag", "let_fnt_scolor_flag", "let_fnteffect_flag",
    "let_alignment_flag", "let_writing_mode_flag", "let_family_flag", "let_autolayout_flag", "let_uppercase_flag",
    "let_show_only_custom_fonts_flag", "let_textstyle_indep_flag", "text_styles_path", "default_text_style_name",
    "fsearch_case", "fsearch_whole_word", "fsearch_regex", "fsearch_range",
    "gsearch_case", "gsearch_whole_word", "gsearch_regex", "gsearch_range",
    "darkmode", "bubbly_ui", "accent_color_hex", "app_font_family", "app_font_size", "use_custom_cursor", "custom_cursor_path", "textselect_mini_menu", "fold_textarea", "show_source_text", "show_trans_text",
    "saladict_shortcut", "search_url", "ocr_sublist", "restore_ocr_empty", "pre_mt_sublist", "mt_sublist",
    "display_lang", "imgsave_quality", "imgsave_webp_lossless", "imgsave_ext", "intermediate_imgsave_ext",
    "enable_glossary_enforcement", "llm_glossary_map", "enable_back_translation_qa", "back_translation_drift_threshold", "translation_prompt_profile_default", "translation_qa_retry_issue_threshold", "llm_token_budget",
    "enable_text_normalization", "text_normalization_profile",
    "runtime_http_timeout_sec", "runtime_http_retries", "data_path_override", "automation_api_enabled", "automation_api_port", "automation_api_key", "credential_use_plaintext_fallback", "credential_migration_done", "automation_api_job_history_limit", "automation_api_job_log_limit",
    "user_replace_profiles",
    "vertical_cjk_rotate_latin", "vertical_cjk_punctuation_hang",
    "pipeline_retry_detect", "pipeline_retry_ocr", "pipeline_retry_translate", "pipeline_retry_inpaint",
    "supersampling_factor", "show_text_style_preset", "expand_tstyle_panel", "show_text_effect_panel",
    "expand_teffect_panel", "text_advanced_format_panel", "expand_tadvanced_panel", "config_panel_font_scale",
    "show_advanced_settings",
    "default_device", "unload_after_idle_minutes", "ocr_spell_check", "spell_check_highlight",
    "manga_source_lang", "manga_source_data_saver", "manga_source_download_dir",
    "manga_source_request_delay", "manga_source_open_after_download", "manga_source_playwright_headless",
    "manga_source_translate_raw_search", "manga_source_show_experimental", "manga_source_base_url_overrides", "manga_source_filter",
    "model_packages_enabled",
    "offline_local_only_mode",
    "show_module_tier_badges_in_tooltips",
    "model_packages_enabled", "model_download_last_status",
    "model_packages_enabled", "model_package_preset_ids",
    "dev_mode",
    "diagnostic_mode",
    "release_caches_after_batch", "manual_mode", "skip_ignored_in_run", "skip_satisfied_pipeline_steps", "auto_mark_translated_pages", "render_default_writing_mode", "render_default_fit_mode", "render_default_line_break_strategy", "render_default_reading_order", "render_overflow_warnings", "render_diagnostics_overlay", "render_auto_polish_on_ocr", "render_default_font_family", "render_default_stroke_width", "render_default_secondary_stroke_width", "render_default_secondary_stroke_color", "render_default_shadow_radius", "render_default_shadow_strength", "render_default_text_padding", "render_atomic_fit_target_fill", "render_atomic_fit_max_expand", "render_atomic_fit_profile", "render_fallback_fonts_latin", "render_fallback_fonts_cjk", "render_fallback_fonts_korean", "render_fallback_fonts_rtl", "render_fallback_fonts_emoji", "render_favorite_fonts", "render_recent_fonts", "render_custom_manga_presets", "export_open_folder_after_batch", "export_include_unrendered_pages", "export_filename_template", "text_editor_top_padding",
    "smooth_scroll_duration_ms", "motion_blur_on_scroll", "reduce_motion",
    "shortcuts", "auto_region_merge_after_run", "region_merge_settings", "context_menu", "context_menu_pinned", "context_run_macros",
    "huggingface_token", "translator_last_model_by_provider",
    "windows_context_menu_offered",
)

def context_menu_visible(key: str) -> bool:
    """Whether the context menu action with this key should be shown. Missing key => True."""
    if not hasattr(pcfg, 'context_menu') or not isinstance(pcfg.context_menu, dict):
        return True
    return pcfg.context_menu.get(key, True)


def diagnostic_logging_enabled() -> bool:
    return bool(getattr(pcfg, 'diagnostic_mode', False))


def log_diagnostic_event(event: str, **payload):
    """Emit one structured diagnostic log line when diagnostic mode is enabled."""
    if not diagnostic_logging_enabled():
        return
    try:
        formatted = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        formatted = str(payload)
    LOGGER.info("DIAG|%s|%s", event, formatted)


def load_textstyle_from(p: str, raise_exception = False):

    if not osp.exists(p):
        LOGGER.warning(f'Text style {p} does not exist.')
        return

    try:
        with open(p, 'r', encoding='utf8') as f:
            style_list = json.loads(f.read())
            styles_loaded = []
            for style in style_list:
                try:
                    styles_loaded.append(FontFormat(**style))
                except Exception as e:
                    LOGGER.warning(f'Skip invalid text style: {style}')
    except Exception as e:
        LOGGER.error(f'Failed to load text style from {p}: {e}')
        if raise_exception:
            raise e
        return

    global text_styles, pcfg
    if len(text_styles) > 0:
        text_styles.clear()
    text_styles.extend(styles_loaded)
    pcfg.text_styles_path = p

def load_config(config_path: str = shared.CONFIG_PATH):
    if config_path != shared.CONFIG_PATH:
        shared.CONFIG_PATH = config_path
        LOGGER.info(f'Using specified config file at {shared.CONFIG_PATH}')

    config_file_existed = osp.exists(shared.CONFIG_PATH)
    if config_file_existed:
        try:
            config = ProgramConfig.load(shared.CONFIG_PATH)
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning("Failed to load config file, using default config")
            config = ProgramConfig()
        shared.FIRST_RUN_NO_CONFIG = False
    if not config_file_existed:
        LOGGER.info(f'{shared.CONFIG_PATH} does not exist, new config file will be created.')
        shared.FIRST_RUN_NO_CONFIG = True
        example_path = osp.join(osp.dirname(shared.CONFIG_PATH), 'config.example.json')
        if osp.isfile(example_path):
            try:
                config = ProgramConfig.load(example_path)
                LOGGER.info(f'Loaded recommended defaults from {example_path}.')
            except Exception as e:
                LOGGER.warning(f'Could not load config.example.json: {e}. Using code defaults.')
                config = ProgramConfig()
        else:
            config = ProgramConfig()
    
    global pcfg
    pcfg.merge(config)
    # Migrate removed rtdetr_comic / legacy rtdetr_v2 -> ctd (rtdetr_comic detector removed)
    for key in ('textdetector', 'textdetector_secondary', 'textdetector_tertiary'):
        if getattr(pcfg.module, key, None) in ('rtdetr_comic', 'rtdetr_v2'):
            setattr(pcfg.module, key, 'ctd')
    tp = getattr(pcfg.module, 'textdetector_params', None)
    if isinstance(tp, dict):
        for old in ('rtdetr_v2', 'rtdetr_comic'):
            tp.pop(old, None)
    # Migrate removed surya_ocr (OCR disabled; use surya_det + other OCR)
    if getattr(pcfg.module, 'ocr', None) == 'surya_ocr':
        pcfg.module.ocr = 'rapidocr'
        op = getattr(pcfg.module, 'ocr_params', None)
        if isinstance(op, dict):
            op.pop('surya_ocr', None)
    # Migrate legacy nemotron_ocr → nemotron_parse (Parse 1.1)
    if getattr(pcfg.module, 'ocr', None) == 'nemotron_ocr':
        pcfg.module.ocr = 'nemotron_parse'
        op = getattr(pcfg.module, 'ocr_params', None)
        if isinstance(op, dict) and 'nemotron_ocr' in op:
            if 'nemotron_parse' not in op:
                op['nemotron_parse'] = op.pop('nemotron_ocr')
            else:
                op.pop('nemotron_ocr', None)
    migrate_secrets_to_keyring_if_possible(pcfg, has_keyring=has_keyring, set_secret=set_secret)
    # Section 9: clamp numeric settings
    try:
        from utils.validation import clamp_settings
        clamp_settings(pcfg)
    except Exception:
        pass
    # Merge context menu visibility with defaults (new keys default to True)
    if hasattr(pcfg, 'context_menu') and isinstance(pcfg.context_menu, dict):
        for k, v in CONTEXT_MENU_DEFAULT.items():
            if k not in pcfg.context_menu:
                pcfg.context_menu[k] = v
    else:
        pcfg.context_menu = dict(CONTEXT_MENU_DEFAULT)
    if not isinstance(getattr(pcfg, 'context_menu_pinned', None), list):
        pcfg.context_menu_pinned = []
    if not isinstance(getattr(pcfg, 'context_run_macros', None), list):
        pcfg.context_run_macros = [
            {'id': 'detect_ocr_translate', 'label': 'Macro: Detect+OCR+Translate', 'mode': 1, 'detect_first': True},
            {'id': 'ocr_translate_inpaint', 'label': 'Macro: OCR+Translate+Inpaint', 'mode': 2, 'detect_first': False},
        ]
    # Ensure all shortcut keys exist (merge defaults for any new action)
    try:
        from utils.shortcuts import get_default_shortcuts
        defaults = get_default_shortcuts()
        if not isinstance(getattr(pcfg, 'shortcuts', None), dict):
            pcfg.shortcuts = dict(defaults)
        else:
            for k, v in defaults.items():
                if k not in pcfg.shortcuts:
                    pcfg.shortcuts[k] = v
    except Exception:
        pass
    # Trim recent projects to configured max
    max_recent = getattr(pcfg, 'recent_proj_list_max', 14)
    if max_recent > 0 and len(pcfg.recent_proj_list) > max_recent:
        pcfg.recent_proj_list = pcfg.recent_proj_list[:max_recent]
    # Substitute empty module device with global default device
    try:
        from modules.base import DEFAULT_DEVICE
        default_dev = (getattr(pcfg, 'default_device', None) or '').strip() or DEFAULT_DEVICE
        for param_key in ('textdetector_params', 'ocr_params', 'translator_params', 'inpainter_params'):
            d = getattr(pcfg.module, param_key, None)
            if not d:
                continue
            for mod_name, mod_params in d.items():
                if not isinstance(mod_params, dict) or 'device' not in mod_params:
                    continue
                dev = mod_params.get('device')
                if isinstance(dev, dict) and (not (dev.get('value') or '').strip()):
                    dev['value'] = default_dev
    except Exception:
        pass

    # Section 8: apply HuggingFace token from config so gated models and Xet use it
    try:
        token = (getattr(pcfg, 'huggingface_token', None) or '').strip()
        if token:
            from utils.model_manager import get_model_manager
            get_model_manager().set_hf_token(token)
    except Exception:
        pass

    p = (pcfg.text_styles_path or '').strip()
    if not p:
        p = osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')
        pcfg.text_styles_path = p
    p = osp.normpath(osp.abspath(p))
    pcfg.text_styles_path = p
    dp = osp.normpath(osp.abspath(osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')))
    if not osp.exists(p):
        if p != dp and osp.exists(dp):
            p = dp
            pcfg.text_styles_path = p
            LOGGER.warning(f'Text style path missing; using default at {dp}.')
        else:
            try:
                os.makedirs(osp.dirname(dp), exist_ok=True)
            except Exception:
                pass
            with open(dp, 'w', encoding='utf8') as f:
                f.write(json.dumps([],  ensure_ascii=False))
            LOGGER.info(f'New text style file created at {dp}.')
            p = dp
            pcfg.text_styles_path = p
    load_textstyle_from(p)

    # Create config.json on first run so ZIP users get recommended defaults from config.example.json
    if not osp.exists(shared.CONFIG_PATH):
        save_config(force=True)


def json_dump_program_config(obj, **kwargs):
    def _default(o):
        if isinstance(o, (np.ndarray, np.ScalarType)):
            return serialize_np(o)
        elif isinstance(o, ModuleConfig):
            return o.get_saving_params()
        elif type(o).__name__ == "ProgramConfig":
            # Section 9: canonical key order for clean diffs
            ordered = {}
            for k in CONFIG_KEY_ORDER:
                if hasattr(o, k):
                    ordered[k] = getattr(o, k)
            for k in o.__dict__:
                if k not in ordered:
                    ordered[k] = getattr(o, k)
            return ordered
        return o.__dict__
    return json.dumps(obj, default=lambda o: _default(o), ensure_ascii=False, **kwargs)


# --- Debounced config saving (reduces log spam / disk writes) ---
_CONFIG_SAVE_DEBOUNCE_SEC = 0.6
_config_save_lock = threading.Lock()
_config_save_timer: Optional[threading.Timer] = None
_config_save_last_write_ts: float = 0.0
_config_save_pending: bool = False


def _flush_debounced_config_save():
    """Timer callback: perform a trailing save if there were pending requests."""
    with _config_save_lock:
        global _config_save_timer, _config_save_pending
        _config_save_timer = None
        pending = bool(_config_save_pending)
        _config_save_pending = False
    if pending:
        try:
            save_config(force=True)
        except Exception:
            pass


def flush_config_save() -> bool:
    """Force an immediate save if one is pending (e.g., before exit)."""
    with _config_save_lock:
        global _config_save_timer, _config_save_pending
        if _config_save_timer is not None:
            try:
                _config_save_timer.cancel()
            except Exception:
                pass
            _config_save_timer = None
        pending = bool(_config_save_pending)
        _config_save_pending = False
    if pending:
        return bool(save_config(force=True))
    return False




def save_config(force: bool = False):
    """Save program config to user config file (config/config.json). Never writes to config.example.json.

    When force=False (default), this is debounced to avoid frequent writes when UI changes many settings quickly.
    """
    global pcfg
    if osp.basename(shared.CONFIG_PATH) == 'config.example.json':
        LOGGER.warning('Refusing to save to config.example.json; user config is config.json (gitignored).')
        return False
    if not force:
        now = time.time()
        with _config_save_lock:
            global _config_save_timer, _config_save_last_write_ts, _config_save_pending
            if (now - (_config_save_last_write_ts or 0.0)) < _CONFIG_SAVE_DEBOUNCE_SEC:
                _config_save_pending = True
                if _config_save_timer is None:
                    delay = max(0.05, _CONFIG_SAVE_DEBOUNCE_SEC - (now - (_config_save_last_write_ts or 0.0)))
                    _config_save_timer = threading.Timer(delay, _flush_debounced_config_save)
                    _config_save_timer.daemon = True
                    _config_save_timer.start()
                return True
    try:
        tmp_save_tgt = shared.CONFIG_PATH + '.tmp'
        with open(tmp_save_tgt, 'w', encoding='utf8') as f:
            f.write(json_dump_program_config(pcfg))
    except Exception as e:
        LOGGER.error(f'Failed save config to {tmp_save_tgt}: {e}')
        LOGGER.error(traceback.format_exc())
        return False

    os.replace(tmp_save_tgt, shared.CONFIG_PATH)
    LOGGER.info('Config saved')
    with _config_save_lock:
        _config_save_last_write_ts = time.time()
        _config_save_pending = False
    return True

def save_text_styles(raise_exception = False):
    global pcfg, text_styles
    try:
        style_dir = osp.dirname(pcfg.text_styles_path)
        if not osp.exists(style_dir):
            os.makedirs(style_dir)
        tmp_save_tgt = pcfg.text_styles_path + '.tmp'
        with open(tmp_save_tgt, 'w', encoding='utf8') as f:
            f.write(json_dump_nested_obj(text_styles))

    except Exception as e:
        LOGGER.error(f'Failed save text style to {tmp_save_tgt}: {e}')
        LOGGER.error(traceback.format_exc())
        if raise_exception:
            raise e
        return False

    os.replace(tmp_save_tgt, pcfg.text_styles_path)
    LOGGER.info('Text style saved')
    return True


def parse_context_run_macros(raw: str):
    import json
    data = json.loads((raw or '').strip() or '[]')
    if not isinstance(data, list):
        raise ValueError('Run macros must be a JSON list.')
    norm = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f'Run macro #{i+1} must be an object.')
        label = str(item.get('label', '')).strip()
        if not label:
            raise ValueError(f'Run macro #{i+1} missing label.')
        mode = int(item.get('mode', 1))
        detect_first = bool(item.get('detect_first', False))
        mid = str(item.get('id', f'macro_{i+1}')).strip()
        norm.append({'id': mid, 'label': label, 'mode': mode, 'detect_first': detect_first})
    return norm
