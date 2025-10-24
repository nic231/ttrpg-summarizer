# Setup Guide for Python 3.11 Virtual Environment

This guide helps you create a fresh Python 3.11 environment with better RTX 5080 GPU support.

## Prerequisites

1. **Install Python 3.11** (if not already installed)
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"
   - Verify: `python3.11 --version`

2. **Ensure you have:**
   - NVIDIA GPU drivers installed
   - CUDA 12.4 or compatible (usually comes with GPU drivers)

## Setup Steps

### 1. Create Python 3.11 Virtual Environment

Open PowerShell in the project directory:

```powershell
cd "E:\Dropbox\Python\TTRPG Sumariser"

# Create new virtual environment with Python 3.11
python3.11 -m venv .venv_py311

# Activate it
.\.venv_py311\Scripts\Activate.ps1
```

If you get a script execution error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install PyTorch with CUDA Support (Critical - Do This First!)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Verify GPU support:**
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

You should see:
- `CUDA available: True`
- `GPU: NVIDIA GeForce RTX 5080`

### 3. Install Other Packages

```powershell
pip install -r requirements_py311.txt
```

### 4. Test Installation

```powershell
# Test imports
python -c "import whisper; import ollama; from pyannote.audio import Pipeline; print('All packages loaded successfully!')"
```

### 5. Run the Summarizer

```powershell
python ttrpg_summarizer.py
```

## Expected Improvements with Python 3.11

✅ **Better package compatibility** - More stable than Python 3.13
✅ **Full GPU support** - Both Whisper AND diarization should use GPU
✅ **Faster diarization** - 3-5 minutes instead of 15-20 minutes for 1hr file
✅ **No kernel errors** - Should avoid RTX 5080 compatibility issues

## Troubleshooting

### Issue: "python3.11 not found"

If you only have one Python installation, try:
```powershell
python -m venv .venv_py311
```

Then check the Python version:
```powershell
.\.venv_py311\Scripts\python.exe --version
```

If it's not 3.11, you'll need to install Python 3.11 first.

### Issue: GPU still not detected

1. Update GPU drivers: https://www.nvidia.com/Download/index.aspx
2. Reinstall PyTorch:
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

### Issue: pyannote.audio errors

If you still get errors with pyannote 3.3.2, try the older stable version:
```powershell
pip install pyannote.audio==3.1.1
```

Then update the code to use `use_auth_token` instead of `token` (already done).

## Switching Between Environments

**Use Python 3.13 environment (current):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Use Python 3.11 environment (new):**
```powershell
.\.venv_py311\Scripts\Activate.ps1
```

## Performance Comparison

| Component | Python 3.13 (CPU Diarization) | Python 3.11 (GPU Diarization) |
|-----------|-------------------------------|-------------------------------|
| Whisper   | GPU ✅ (~15 min)              | GPU ✅ (~15 min)              |
| Diarization| CPU ❌ (~20 min)             | GPU ✅ (~3-5 min)             |
| Ollama    | GPU ✅ (~10 min)              | GPU ✅ (~10 min)              |
| **Total** | **~45 min**                   | **~30 min**                   |

## Notes

- Keep your Python 3.13 environment - don't delete it!
- You can switch between them to compare performance
- The Python 3.11 environment is specifically for GPU acceleration

## Need Help?

Check the main README.md or the code comments in ttrpg_summarizer.py
