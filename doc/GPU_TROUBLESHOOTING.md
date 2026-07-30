# GPU / CUDA troubleshooting

If CUDA is installed on the machine but the app only shows `cpu` in module device selectors, the Python environment is usually using a CPU-only PyTorch wheel.

## Automatic launcher repair

When started through `launch.py`, BallonsTranslator now checks for an NVIDIA GPU before importing the main app. If an NVIDIA GPU is visible and the installed PyTorch build is CPU-only or cannot use CUDA, the launcher reinstalls the CUDA-enabled PyTorch wheel so `cuda` appears in device selectors after restart.

Set `BT_SKIP_CUDA_TORCH_REINSTALL=1` if you intentionally want to keep a CPU-only PyTorch install.

## Manual checks

Run:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected for NVIDIA GPU use:

- `torch.version.cuda` is not `None`.
- `torch.cuda.is_available()` is `True`.
- `torch.cuda.device_count()` is at least `1`.

If those checks fail, reinstall PyTorch with the command printed by the launcher or run:

```bash
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then restart the app and check **Config → General → Device diagnostics**.

## "No matching distribution found for torch" on a very new Python

The pinned CUDA wheels (`cu118`) are only published for CPython 3.9–3.13. On a newer
interpreter (for example 3.14) the launcher drops the version pin and uses the latest CUDA
index instead; if that index has no wheel either, the launcher installs the CPU wheels and
starts without GPU acceleration rather than aborting.

To get GPU acceleration on such a setup, either install a supported Python version (3.12 is
recommended) or point the launcher at a working command:

```bash
set TORCH_COMMAND=pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
