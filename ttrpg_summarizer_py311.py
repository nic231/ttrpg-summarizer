"""
TTRPG Session Summarizer - Python 3.11 Optimized Version
Converts audio recordings to text and generates summaries using Ollama
GPU-ENABLED SPEAKER DIARIZATION and TRANSCRIPTION for better performance!

Features:
    - Audio transcription using OpenAI Whisper (local GPU)
    - Speaker diarization with PyAnnote Audio
    - AI summarization using Ollama (qwen2.5:14b)
    - Campaign character tracking across sessions
    - VAD for multi-file mode efficiency
    - Progressive context building for chunk summaries

Requirements:
    Python 3.11
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements_py311.txt

Author: TTRPG Summarizer Team
Version: 1.1.0
"""

__version__ = "1.1.0"

import sys
import json
import warnings
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import whisper
import ollama
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pyannote.audio import Pipeline

# Try importing faster-whisper for VAD support
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not installed. VAD will not be available.")

# Verify CUDA setup at startup
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='torchcodec is not installed correctly')
warnings.filterwarnings('ignore', category=UserWarning, module='pyannote')


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Configuration file paths
CONFIG_DIR = Path.home() / ".ttrpg_summarizer"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Default directories (relative to script location)
SCRIPT_DIR = Path(__file__).parent if Path(__file__).exists() else Path.cwd()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "summaries"
DEFAULT_NPCS_DIR = SCRIPT_DIR / "NPCs"

# Processing constants
WORDS_PER_MINUTE = 150  # Average speaking rate
DEFAULT_CHUNK_SIZE_WORDS = 4000  # Words per chunk
DEFAULT_CHUNK_OVERLAP_WORDS = 200  # Overlap between chunks
DEFAULT_TARGET_WORDS_CHUNK = 400  # Target words for chunk summary
DEFAULT_TARGET_WORDS_FINAL = 1200  # Target words for final summary

# Context window sizes
DEFAULT_CONTEXT_SIZE = 8192  # Default tokens for context window
MAX_CONTEXT_SIZE = 131072  # Maximum supported context size
UNLIMITED_CONTEXT = 999999  # Special value meaning "no limit"
CONTEXT_INCREMENT = 2048  # Step size for context adjustments

# VRAM estimation constants
VRAM_CONTEXT_MULTIPLIER = 0.041  # GB per 1K tokens per billion parameters
BYTES_PER_GB = 1024 ** 3  # Bytes in a gigabyte

# UI Constants
BANNER_SINGLE = "=" * 60
BANNER_DOUBLE = "=" * 80
SEPARATOR = "-" * 60

# Model defaults (for reference - actual defaults set in GUI initialization)
DEFAULT_WHISPER_MODEL = "medium.en"  # Fast English-only model
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"  # Chunk and final summary model
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"  # Speaker identification

# File extensions
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_timestamp(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS or MM:SS

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins}:{secs:02d}"


def format_time(seconds: float) -> str:
    """
    Format seconds as Hh MMm SSs or MMm SSs

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {mins:02d}m {secs:02d}s" if hours > 0 else f"{mins}m {secs:02d}s"


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================


def load_config() -> Dict:
    """
    Load configuration from file

    Returns:
        Dictionary with configuration (empty if file doesn't exist)
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    return {}


def save_config(config: Dict) -> None:
    """
    Save configuration to file

    Args:
        config: Dictionary with configuration to save
    """
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {CONFIG_FILE}")
    except Exception as e:
        print(f"Warning: Could not save config: {e}")


def get_saved_token() -> Optional[str]:
    """
    Get saved HuggingFace token from config

    Returns:
        Token string if saved, None otherwise
    """
    config = load_config()
    return config.get("hf_token")


def save_token(token: str) -> None:
    """
    Save HuggingFace token to config

    Args:
        token: HuggingFace access token
    """
    config = load_config()
    config["hf_token"] = token
    save_config(config)


def clear_token() -> None:
    """
    Remove saved HuggingFace token from config
    """
    config = load_config()
    if "hf_token" in config:
        del config["hf_token"]
        save_config(config)
        print("Saved token cleared")


def check_ollama_running() -> bool:
    """
    Check if Ollama is running and accessible

    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        ollama.list()
        return True
    except Exception:
        return False


def prompt_start_ollama() -> bool:
    """
    Prompt user to start Ollama if it's not running

    Returns:
        True if user wants to continue (after starting Ollama), False to exit
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    result = messagebox.askretrycancel(
        "Ollama Not Running",
        "Ollama is not running. Please start Ollama and click 'Retry'.\n\n"
        "To start Ollama:\n"
        "- Open Ollama application, or\n"
        "- Run 'ollama serve' in a terminal\n\n"
        "Click 'Cancel' to exit."
    )

    root.destroy()
    return result


def get_available_vram() -> tuple:
    """
    Get available VRAM information

    Returns:
        Tuple of (total_vram_gb, available_vram_gb, gpu_name) or (0, 0, "CPU") if no GPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Get actual free memory from GPU driver
            torch.cuda.empty_cache()
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            free_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            return (total_gb, free_gb, gpu_name)
        else:
            return (0, 0, "CPU")
    except (RuntimeError, ImportError, subprocess.SubprocessError) as e:
        print(f"Warning: Failed to get VRAM info: {e}")
        return (0, 0, "Unknown")


class TTRPGSummarizer:
    def __init__(self, whisper_model: str = "base", ollama_model: str = "llama3.2",
                 enable_diarization: bool = False, hf_token: Optional[str] = None,
                 chunk_target_words: int = DEFAULT_TARGET_WORDS_CHUNK,
                 final_summary_target_words: int = DEFAULT_TARGET_WORDS_FINAL,
                 chunk_context_size: int = DEFAULT_CONTEXT_SIZE, ollama_final_model: Optional[str] = None,
                 final_context_size: int = None,
                 stop_event=None, pause_event=None, skip_whisper_load: bool = False,
                 campaign_character_file: Optional[str] = None,
                 preconfigured_final_summary: bool = False):
        """
        Initialize the TTRPG Summarizer

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            ollama_model: Ollama model to use for chunk summarization
            enable_diarization: Enable speaker diarization (requires HuggingFace token)
            hf_token: HuggingFace API token for speaker diarization
            campaign_character_file: Path to campaign character tracking file (optional)
            chunk_target_words: Target word count for chunk summaries (default: 400)
            final_summary_target_words: Target word count for final summary (default: 1200)
            chunk_context_size: Context window size for chunk summaries (default: 8192)
            ollama_final_model: Separate Ollama model for final summary (default: same as chunk model)
            skip_whisper_load: Skip loading Whisper model (for JSON-only processing)
        """
        # Detect GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        print(f"Using device: {device_name} ({self.device})")

        # Store model name for faster-whisper fallback
        self.whisper_model_name = whisper_model

        if skip_whisper_load:
            print("Skipping Whisper model loading (JSON-only mode)...")
        else:
            print(f"Loading Whisper model '{whisper_model}'...")
            # Try loading directly to GPU for Python 3.11
            try:
                self.whisper_model = whisper.load_model(whisper_model, device=self.device)
                print(f"Whisper model loaded successfully on {self.device.upper()}!")
            except Exception as e:
                print(f"Failed to load on GPU, falling back to CPU: {e}")
                self.whisper_model = whisper.load_model(whisper_model, device="cpu")
                if self.device == "cuda":
                    self.whisper_model = self.whisper_model.to(self.device)
                print("Whisper model loaded successfully!")

        self.ollama_model = ollama_model
        self.ollama_final_model = ollama_final_model or ollama_model  # Use same model if not specified
        self.chunk_target_words = chunk_target_words
        self.final_summary_target_words = final_summary_target_words
        self.chunk_context_size = chunk_context_size
        self.final_context_size = final_context_size  # None or UNLIMITED_CONTEXT = unlimited
        self.preconfigured_final_summary = preconfigured_final_summary  # Skip popup if pre-configured

        # Control events for pause/stop
        self.stop_event = stop_event
        self.pause_event = pause_event

        # Character/NPC tracking
        self.campaign_character_file = campaign_character_file
        self.characters = {}  # Dict: {name: {first_appearance: session_name, description: str, mentions: count}}
        self.current_session_name = None  # Track current session being processed

        # Load existing characters from campaign file if provided
        if campaign_character_file and Path(campaign_character_file).exists():
            self.load_campaign_characters()

        # Speaker diarization setup - GPU ENABLED with PyTorch 2.7.0!
        self.enable_diarization = enable_diarization
        self.diarization_pipeline = None

        if enable_diarization:
            if not hf_token:
                raise ValueError("HuggingFace token required for speaker diarization")
            print("Loading speaker diarization model...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

            # Move to GPU with PyTorch 2.7.0 + CUDA 12.8 (RTX 5080 supported!)
            if self.device == "cuda":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                print("✓ Speaker diarization model loaded successfully on GPU!")
            else:
                print("Speaker diarization model loaded successfully (running on CPU)")

    def unload_models(self):
        """Unload all models from VRAM/memory"""
        print("\n🧹 Unloading models from memory...")

        # Unload Whisper model
        if hasattr(self, 'whisper_model'):
            del self.whisper_model
            print("  ✓ Whisper model unloaded")

        # Unload diarization pipeline
        if hasattr(self, 'diarization_pipeline') and self.diarization_pipeline is not None:
            del self.diarization_pipeline
            print("  ✓ Diarization pipeline unloaded")

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("  ✓ GPU memory cleared")

        print("✓ All models unloaded successfully!\n")

    def check_control_events(self):
        """Check if processing should stop or pause"""
        if self.stop_event and self.stop_event.is_set():
            print("\n⛔ STOPPED by user")
            self.unload_models()
            raise KeyboardInterrupt("Processing stopped by user")

        if self.pause_event:
            while not self.pause_event.is_set():
                # Wait while paused
                import time
                time.sleep(0.1)
                # Check for stop during pause
                if self.stop_event and self.stop_event.is_set():
                    print("\n⛔ STOPPED by user")
                    self.unload_models()
                    raise KeyboardInterrupt("Processing stopped by user")

    def transcribe_audio(self, audio_file: str, use_vad: bool = False) -> Dict:
        """
        Convert audio file to text using Whisper

        Args:
            audio_file: Path to audio file (mp3, wav, m4a, etc.)
            use_vad: Enable Voice Activity Detection to skip silence (useful for multi-file mode)

        Returns:
            Dictionary with transcription and metadata
        """
        print(f"\n{BANNER_SINGLE}")
        print(f"AUDIO TRANSCRIPTION")
        print(f"{BANNER_SINGLE}")
        print(f"File: {Path(audio_file).name}")
        print(f"Model: whisper-{self.whisper_model.__class__.__name__}")
        print(f"{BANNER_SINGLE}\n")

        # Convert to WAV for optimal GPU performance (if not already WAV)
        audio_path, is_temp = self.convert_to_wav_if_needed(audio_file)

        import time
        import sys

        # Get audio duration for progress calculation (use converted file)
        try:
            import librosa
            audio_duration = librosa.get_duration(path=audio_path)
            # Estimate word count: average speaking rate is ~150 words/minute
            estimated_words = int(audio_duration / 60 * WORDS_PER_MINUTE)
            print(f"Audio duration: {audio_duration/60:.1f} minutes")
            print(f"Estimated words: ~{estimated_words:,} words\n")
        except Exception as e:
            print(f"Warning: Could not estimate audio duration: {e}")
            audio_duration = None
            estimated_words = None

        start_time = time.time()

        # Use custom writer to filter verbose output (show progress, hide transcript lines)
        print("Transcribing audio...\n")

        # Create a custom writer that filters out transcript lines but keeps progress
        import sys
        class WhisperProgressFilter:
            """Filter Whisper verbose output to show only progress bars, not transcript text"""
            def __init__(self, original_stdout):
                self.original_stdout = original_stdout
                self.buffer = ""

            def write(self, text):
                # Progress bars contain %| pattern, transcript lines start with [timestamp]
                if '%|' in text:
                    # This is a progress bar - show it
                    self.original_stdout.write(text)
                elif text.strip() and not text.strip().startswith('['):
                    # Non-timestamp text (errors, warnings) - show it
                    self.original_stdout.write(text)
                # Else: transcript line with [timestamp] - filter it out

            def flush(self):
                self.original_stdout.flush()

        # Temporarily redirect stdout to filter verbose output
        original_stdout = sys.stdout
        sys.stdout = WhisperProgressFilter(original_stdout)

        try:
            # Build transcription parameters
            transcribe_params = {
                "audio": audio_path,  # Use converted WAV file for better GPU performance
                "language": "en",
                "verbose": True,  # Enable verbose but filter transcript lines with custom writer
                "fp16": (self.device == "cuda"),
                "condition_on_previous_text": False  # Prevent hallucinations from context
            }

            # Use faster-whisper for ALL transcriptions if available (4x faster + better accuracy)
            # Enable VAD in multi-file mode to skip muted microphone sections
            if FASTER_WHISPER_AVAILABLE:
                if use_vad:
                    print("  Using faster-whisper with VAD (silero-vad) to skip muted sections\n")
                else:
                    print("  Using faster-whisper (4x faster than standard Whisper)\n")

                # Load faster-whisper model
                faster_model = FasterWhisperModel(
                    self.whisper_model_name,
                    device="cuda" if self.device == "cuda" else "cpu",
                    compute_type="float16" if self.device == "cuda" else "int8"
                )

                # Transcribe with optional VAD
                segments_gen, info = faster_model.transcribe(
                    audio_path,
                    language="en",
                    vad_filter=use_vad,  # Enable VAD only for multi-file mode
                    vad_parameters=dict(
                        min_silence_duration_ms=2000,  # 2 seconds of silence to skip
                        threshold=0.5  # Sensitivity
                    ) if use_vad else None,
                    condition_on_previous_text=False
                )

                # Convert generator to list and format like standard whisper
                segments_list = list(segments_gen)
                result = {
                    "text": " ".join(seg.text for seg in segments_list),
                    "segments": [
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text
                        }
                        for seg in segments_list
                    ],
                    "language": info.language
                }

                # Clean up faster-whisper model
                del faster_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Fallback to standard Whisper
                print("  ⚠️  faster-whisper not available, using standard Whisper")
                print("  Run: pip install faster-whisper for 4x speedup\n")
                result = self.whisper_model.transcribe(**transcribe_params)
        except Exception as e:
            # Restore stdout before re-raising
            sys.stdout = original_stdout
            raise e
        finally:
            # Restore original stdout
            sys.stdout = original_stdout

        # DON'T clean up temp WAV yet - it's needed for speaker audio clip extraction later
        # Cleanup will happen after speaker identification is complete

        elapsed = time.time() - start_time
        print(f"\n✓ Transcription complete in {elapsed/60:.1f} minutes!")
        print(f"  Total text length: {len(result['text']):,} characters")

        # Filter out hallucinated repetitions from segments
        print("Filtering hallucinated repetitions...")
        raw_segments = result["segments"]  # Keep original segments
        cleaned_segments = self.filter_hallucinations(raw_segments)
        removed_count = len(raw_segments) - len(cleaned_segments)
        if removed_count > 0:
            print(f"  Removed {removed_count} repetitive/hallucinated segments")

        # Rebuild full text from cleaned segments
        cleaned_text = " ".join(seg["text"].strip() for seg in cleaned_segments)

        transcription_data = {
            "text": cleaned_text,
            "segments": cleaned_segments,  # Timestamped segments (filtered)
            "raw_segments": raw_segments,  # Original unfiltered segments
            "language": result["language"],
            "audio_file": audio_file,
            "timestamp": datetime.now().isoformat(),
            "temp_wav_file": audio_path if is_temp else None  # Track temp WAV for cleanup later
        }

        return transcription_data

    def filter_hallucinations(self, segments: List[Dict]) -> List[Dict]:
        """
        Remove segments that are likely hallucinations (excessive repetition)

        Whisper is known to hallucinate by repeating phrases, especially in:
        - Silent sections
        - Background noise
        - Unclear audio

        This filter catches:
        1. Consecutive identical segments (immediate repetition)
        2. Longer phrases (5+ words) repeated 2+ times in last 5 segments
        3. Very short phrases (1-2 words) repeated 3+ times consecutively

        Args:
            segments: List of transcription segments

        Returns:
            Filtered list without obvious hallucinations
        """
        if not segments:
            return segments

        cleaned = []
        recent_texts = []  # Track last 10 segments
        prev_text = None   # Track immediate previous segment
        consecutive_count = 0  # Track consecutive identical phrases

        for seg in segments:
            text = seg["text"].strip().lower()

            # Skip empty segments
            if not text or len(text.split()) == 0:
                continue

            word_count = len(text.split())

            # Check for consecutive identical segments
            if prev_text and text == prev_text:
                consecutive_count += 1
                # For very short phrases (1-2 words), allow 2 repetitions (3 total)
                # For longer phrases, skip immediately on first repeat
                if word_count <= 2:
                    if consecutive_count >= 3:
                        continue  # Skip the 4th+ repetition
                else:
                    continue  # Skip any consecutive repeat of 3+ word phrases
            else:
                consecutive_count = 1  # Reset counter

            # For longer phrases (5+ words), check if repeated in recent history
            if word_count >= 5:
                if recent_texts.count(text) >= 2:
                    # Longer phrase appeared 2+ times in last 10 segments - likely hallucination
                    continue

            # Keep this segment
            cleaned.append(seg)
            prev_text = text

            # Update recent history (increased to 10 segments for better context)
            recent_texts.append(text)
            if len(recent_texts) > 10:
                recent_texts.pop(0)

        return cleaned

    def convert_to_wav_if_needed(self, audio_file: str) -> tuple[str, bool]:
        """
        Convert audio file to WAV with audio normalization for optimal Whisper transcription

        Applies:
        - Audio normalization (loudnorm) to boost quiet audio
        - Converts to 16kHz mono WAV for optimal GPU performance
        - Normalizes to -16 LUFS (speech standard)

        Args:
            audio_file: Path to audio file

        Returns:
            Tuple of (path_to_audio, is_temp_file)
        """
        file_ext = Path(audio_file).suffix.lower()

        # Always convert to 16kHz mono WAV with normalization for optimal GPU performance
        # - WAV = raw PCM data, no decoding overhead = better GPU utilization
        # - loudnorm = boosts quiet audio so Whisper can transcribe it properly
        # - 16kHz = optimal for speech (Whisper's training rate)
        # Exception: Skip conversion if already a 16kHz mono WAV
        if file_ext != '.wav':
            print(f"Converting {file_ext} to 16kHz mono WAV for optimal GPU performance...")

            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()

            # Use ffmpeg to convert (Whisper already has ffmpeg dependency)
            # Skip loudnorm for now - it's slow (2-pass analysis) and VAD handles silence better
            # Can re-enable if needed, but faster-whisper's VAD makes it less critical
            try:
                # Fast conversion: Just downsample to 16kHz mono
                # loudnorm disabled for speed - VAD handles quiet sections
                subprocess.run(
                    ['ffmpeg', '-i', audio_file,
                     '-ar', '16000', '-ac', '1', '-y', temp_wav_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✓ Converted to WAV: {temp_wav_path}")
                return temp_wav_path, True
            except subprocess.CalledProcessError as e:
                print(f"Error converting audio: {e.stderr}")
                raise
            except FileNotFoundError:
                raise RuntimeError(
                    "ffmpeg not found. Please install ffmpeg:\n"
                    "  Windows: winget install ffmpeg\n"
                    "  Or download from: https://ffmpeg.org/download.html"
                )

        # File format is supported, use as-is
        return str(Path(audio_file).resolve()), False

    def perform_diarization(self, audio_file: str) -> Dict:
        """
        Perform speaker diarization to identify who spoke when

        Args:
            audio_file: Path to audio file

        Returns:
            Dictionary with speaker timeline information
        """
        if not self.enable_diarization or not self.diarization_pipeline:
            raise ValueError("Diarization not enabled. Set enable_diarization=True")

        print(f"\n{'='*60}")
        print(f"SPEAKER DIARIZATION")
        print(f"{'='*60}")
        print(f"File: {Path(audio_file).name}")
        print(f"Processing with GPU acceleration (if available)...")
        print(f"{'='*60}\n")

        # Convert to WAV if needed (pyannote doesn't support m4a/mp3)
        audio_path, is_temp = self.convert_to_wav_if_needed(audio_file)

        import time
        import threading

        start_time = time.time()
        processing_complete = threading.Event()

        def progress_monitor():
            """Display elapsed time every 30 seconds"""
            while not processing_complete.is_set():
                if processing_complete.wait(30):  # Wait 30 seconds or until complete
                    break
                elapsed = time.time() - start_time
                print(f"  [{elapsed/60:.1f} minutes elapsed...]", flush=True)

        # Start progress monitoring in background
        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()

        # Ask user for expected number of speakers (helps reduce over-segmentation)
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        num_speakers_str = simpledialog.askstring(
            "Speaker Count",
            "How many speakers do you expect in this audio?\n\n"
            "Enter a number (e.g., 4-6 for 'between 4 and 6')\n"
            "Or leave blank to auto-detect (may over-segment)",
            parent=root
        )
        root.destroy()

        # Parse speaker count input
        min_speakers = None
        max_speakers = None

        if num_speakers_str and num_speakers_str.strip():
            try:
                if '-' in num_speakers_str:
                    # Range like "4-6"
                    parts = num_speakers_str.split('-')
                    min_speakers = int(parts[0].strip())
                    max_speakers = int(parts[1].strip())
                    print(f"Constraining to {min_speakers}-{max_speakers} speakers")
                else:
                    # Exact number like "5"
                    exact = int(num_speakers_str.strip())
                    min_speakers = exact
                    max_speakers = exact
                    print(f"Constraining to exactly {exact} speakers")
            except (ValueError, TypeError):
                print("Invalid input - using auto-detection")

        # pyannote.audio with batch_size=1 to avoid tensor size mismatch
        try:
            # Try with default batching first
            if min_speakers and max_speakers:
                diarization = self.diarization_pipeline(
                    audio_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
            else:
                diarization = self.diarization_pipeline(audio_path)
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                elapsed_before_retry = time.time() - start_time
                print(f"\nBatching error detected after {elapsed_before_retry/60:.1f} minutes")
                print("Retrying with batch_size=1 (slower but more reliable)...")
                # Set batch_size on the pipeline itself (not the _embedding object)
                original_batch_size = self.diarization_pipeline.embedding_batch_size
                self.diarization_pipeline.embedding_batch_size = 1
                try:
                    # Retry with no batching (slower but more reliable)
                    if min_speakers and max_speakers:
                        diarization = self.diarization_pipeline(
                            audio_path,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers
                        )
                    else:
                        diarization = self.diarization_pipeline(audio_path)
                finally:
                    # Restore original batch size
                    self.diarization_pipeline.embedding_batch_size = original_batch_size
            else:
                processing_complete.set()  # Stop progress monitor
                # Don't clean up yet - might need for retries
                raise
        except Exception as e:
            processing_complete.set()  # Stop progress monitor
            elapsed = time.time() - start_time
            print(f"\n✗ Diarization failed after {elapsed/60:.1f} minutes")
            # Clean up temp WAV on failure (won't be used for clips)
            if is_temp:
                Path(audio_path).unlink(missing_ok=True)
            raise
        finally:
            processing_complete.set()  # Ensure monitor stops

        elapsed = time.time() - start_time
        print(f"\n✓ Diarization completed successfully in {elapsed/60:.1f} minutes")

        # DON'T clean up temp WAV yet - it's needed for speaker audio clip extraction later
        # (Will be cleaned up after speaker identification completes)

        # Convert to a more usable format
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        # Get unique speakers
        unique_speakers = sorted(set(seg["speaker"] for seg in speaker_segments))
        print(f"Detected {len(unique_speakers)} unique speakers: {', '.join(unique_speakers)}")

        return {
            "segments": speaker_segments,
            "speakers": unique_speakers,
            "wav_file": audio_path,  # Include WAV path for fast clip extraction
            "is_temp_wav": is_temp  # Track if WAV should be cleaned up
        }

    def align_diarization_with_transcription(self, transcription_data: Dict,
                                            diarization_data: Dict) -> List[Dict]:
        """
        Align speaker labels with transcription segments

        Args:
            transcription_data: Output from transcribe_audio()
            diarization_data: Output from perform_diarization()

        Returns:
            List of segments with both text and speaker labels
        """
        print("Aligning speakers with transcription...")

        aligned_segments = []

        for trans_seg in transcription_data["segments"]:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            trans_mid = (trans_start + trans_end) / 2

            # Find the speaker at the midpoint of this transcription segment
            speaker = "UNKNOWN"
            for diar_seg in diarization_data["segments"]:
                if diar_seg["start"] <= trans_mid <= diar_seg["end"]:
                    speaker = diar_seg["speaker"]
                    break

            aligned_segments.append({
                "start": trans_start,
                "end": trans_end,
                "text": trans_seg["text"],
                "speaker": speaker
            })

        print(f"Aligned {len(aligned_segments)} segments with speaker labels")
        return aligned_segments

    def fix_unknown_speakers(self, segments: List[Dict]) -> List[Dict]:
        """
        Fix UNKNOWN speaker labels by merging with surrounding speakers

        Args:
            segments: List of segments with speaker labels

        Returns:
            Cleaned segments with UNKNOWN merged into adjacent speakers
        """
        if not segments:
            return segments

        cleaned = []

        for i, seg in enumerate(segments):
            # If this is UNKNOWN, try to assign it to a nearby speaker
            if seg.get("speaker") == "UNKNOWN":
                # Look at previous and next speakers
                prev_speaker = None
                next_speaker = None

                # Find previous non-UNKNOWN speaker
                for j in range(i - 1, -1, -1):
                    if segments[j].get("speaker") != "UNKNOWN":
                        prev_speaker = segments[j]["speaker"]
                        break

                # Find next non-UNKNOWN speaker
                for j in range(i + 1, len(segments)):
                    if segments[j].get("speaker") != "UNKNOWN":
                        next_speaker = segments[j]["speaker"]
                        break

                # Decision logic
                if prev_speaker and next_speaker:
                    # If sandwiched between same speaker, assign to that speaker
                    if prev_speaker == next_speaker:
                        seg["speaker"] = prev_speaker
                    # If between different speakers, check timing
                    elif i > 0:
                        # Calculate time gap from previous
                        prev_gap = seg["start"] - segments[i-1]["end"]
                        # If very close to previous (< 2 seconds), likely same speaker
                        if prev_gap < 2.0:
                            seg["speaker"] = prev_speaker
                        else:
                            seg["speaker"] = next_speaker
                    else:
                        seg["speaker"] = next_speaker
                elif prev_speaker:
                    # Only previous speaker exists
                    seg["speaker"] = prev_speaker
                elif next_speaker:
                    # Only next speaker exists
                    seg["speaker"] = next_speaker
                # else: remains UNKNOWN (shouldn't happen often)

            cleaned.append(seg)

        return cleaned


    def extract_audio_clip(self, audio_file: str, start_time: float, end_time: float, output_file: str):
        """
        Extract a short audio clip from the original audio file

        Args:
            audio_file: Path to original audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_file: Path to save the clip
        """
        import soundfile as sf
        import librosa

        # Load audio segment
        y, sr = librosa.load(audio_file, sr=None, offset=start_time, duration=end_time - start_time)

        # Save as WAV
        sf.write(output_file, y, sr)

    def speaker_identification_dialog(self, speaker_id: str, quotes: List[str], audio_clips: List[str], previous_speaker_data: Dict[str, Dict] = None) -> str:
        """
        Custom dialog for speaker identification with audio playback buttons

        Args:
            speaker_id: ID of the speaker
            quotes: List of text quotes
            audio_clips: List of audio clip file paths
            previous_speaker_data: Dict mapping speaker names to their data (quotes and clips)

        Returns:
            Speaker name entered by user
        """
        import tkinter as tk
        from tkinter import ttk
        import pygame

        # Initialize pygame mixer for audio playback
        pygame.mixer.init()

        result = [None]  # Use list to store result from nested function
        previous_speaker_data = previous_speaker_data or {}

        def play_clip(clip_path):
            """Play an audio clip"""
            try:
                pygame.mixer.music.load(clip_path)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"Error playing audio: {e}")

        def on_submit():
            """Handle submit button"""
            result[0] = name_entry.get()
            dialog.destroy()

        def on_dropdown_select(event):
            """Handle dropdown selection"""
            selected = speaker_dropdown.get()
            if selected and selected != "-- Select Previous Speaker --":
                name_entry.delete(0, tk.END)
                name_entry.insert(0, selected)

        def open_comparison_window():
            """Open window to compare with previous speakers"""
            if not previous_speaker_data:
                return

            comp_window = tk.Toplevel(dialog)
            comp_window.title("Compare with Previous Speakers")
            comp_window.geometry("800x600")
            comp_window.attributes('-topmost', True)

            # Header
            header = tk.Frame(comp_window, bg="#34495e", padx=10, pady=10)
            header.pack(fill=tk.X)
            tk.Label(header, text="Previously Identified Speakers", font=("Arial", 14, "bold"),
                     bg="#34495e", fg="white").pack()
            tk.Label(header, text="Click play buttons to hear their voices",
                     font=("Arial", 9), bg="#34495e", fg="#ecf0f1").pack()

            # Scrollable frame
            canvas = tk.Canvas(comp_window, bg="white")
            scrollbar = ttk.Scrollbar(comp_window, orient="vertical", command=canvas.yview)
            scrollable = tk.Frame(canvas, bg="white")

            scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Add each previous speaker
            for speaker_name, data in previous_speaker_data.items():
                speaker_frame = tk.Frame(scrollable, bg="#ecf0f1", padx=15, pady=10)
                speaker_frame.pack(fill=tk.X, padx=5, pady=5)

                # Speaker name header
                tk.Label(speaker_frame, text=speaker_name, font=("Arial", 12, "bold"),
                         bg="#ecf0f1", fg="#2c3e50").pack(anchor=tk.W, pady=(0, 5))

                # Show first 3 quotes with play buttons
                for idx, (quote, clip_path) in enumerate(zip(data['quotes'][:3], data['clips'][:3])):
                    quote_frame = tk.Frame(speaker_frame, bg="white", pady=3)
                    quote_frame.pack(fill=tk.X, pady=2)

                    play_btn = tk.Button(
                        quote_frame,
                        text="▶️",
                        command=lambda c=clip_path: play_clip(c),
                        bg="#3498db",
                        fg="white",
                        font=("Arial", 8),
                        width=3
                    )
                    play_btn.pack(side=tk.LEFT, padx=(0, 8))

                    quote_label = tk.Label(
                        quote_frame,
                        text=quote,
                        wraplength=650,
                        justify=tk.LEFT,
                        bg="white",
                        font=("Arial", 8)
                    )
                    quote_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

                # Separator between speakers
                tk.Frame(scrollable, height=2, bg="#bdc3c7").pack(fill=tk.X, padx=5, pady=3)

            # Close button
            close_btn = tk.Button(
                comp_window,
                text="Close",
                command=comp_window.destroy,
                bg="#95a5a6",
                fg="white",
                font=("Arial", 10, "bold"),
                width=15
            )
            close_btn.pack(pady=10)

        # Calculate window height based on number of quotes (max 800px, min 600px)
        # Each quote ~70px, header ~120px, input ~250px (increased for better visibility)
        num_quotes = len(quotes)
        calculated_height = max(min(120 + (num_quotes * 70) + 250, 800), 600)

        # Create dialog window - increased width and height for better layout
        dialog = tk.Toplevel()
        dialog.title(f"Identify Speaker: {speaker_id}")
        dialog.geometry(f"850x{calculated_height}")
        dialog.attributes('-topmost', True)
        dialog.grab_set()  # Make modal

        # Header
        header_frame = tk.Frame(dialog, bg="#2c3e50", padx=10, pady=10)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text=f"Who is {speaker_id}?", font=("Arial", 16, "bold"),
                 bg="#2c3e50", fg="white").pack()
        tk.Label(header_frame, text="Listen to the audio samples and enter the speaker's name",
                 font=("Arial", 10), bg="#2c3e50", fg="#ecf0f1").pack()

        # Scrollable frame for quotes
        canvas_frame = tk.Frame(dialog)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(canvas_frame, bg="white")
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add quotes with play buttons
        for idx, (quote, clip_path) in enumerate(zip(quotes, audio_clips)):
            quote_frame = tk.Frame(scrollable_frame, bg="white", pady=5)
            quote_frame.pack(fill=tk.X, padx=5, pady=3)

            # Play button
            play_btn = tk.Button(
                quote_frame,
                text="▶️ Play",
                command=lambda c=clip_path: play_clip(c),
                bg="#3498db",
                fg="white",
                font=("Arial", 9, "bold"),
                cursor="hand2",
                width=8
            )
            play_btn.pack(side=tk.LEFT, padx=(0, 10))

            # Quote text
            quote_label = tk.Label(
                quote_frame,
                text=quote,
                wraplength=550,
                justify=tk.LEFT,
                bg="white",
                font=("Arial", 9)
            )
            quote_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Add separator
            if idx < len(quotes) - 1:
                tk.Frame(scrollable_frame, height=1, bg="#bdc3c7").pack(fill=tk.X, padx=5)

        # Input frame
        input_frame = tk.Frame(dialog, bg="#ecf0f1", padx=20, pady=15)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Dropdown for previous speakers (if any)
        if previous_speaker_data:
            # Compare button
            compare_btn = tk.Button(
                input_frame,
                text="🔊 Compare with Previous Speakers",
                command=open_comparison_window,
                bg="#9b59b6",
                fg="white",
                font=("Arial", 10, "bold"),
                cursor="hand2"
            )
            compare_btn.pack(fill=tk.X, pady=(0, 10))

            tk.Label(input_frame, text="Or select a previous speaker:", font=("Arial", 10), bg="#ecf0f1", fg="#7f8c8d").pack(anchor=tk.W, pady=(0, 5))

            speaker_names = list(previous_speaker_data.keys())
            speaker_dropdown = ttk.Combobox(input_frame, font=("Arial", 11), state="readonly", width=38)
            speaker_dropdown['values'] = ["-- Select Previous Speaker --"] + speaker_names
            speaker_dropdown.current(0)
            speaker_dropdown.pack(fill=tk.X, pady=(0, 10))
            speaker_dropdown.bind('<<ComboboxSelected>>', on_dropdown_select)

        tk.Label(input_frame, text="Enter speaker name:", font=("Arial", 11), bg="#ecf0f1").pack(anchor=tk.W)

        # Pre-fill with speaker ID (e.g., "SPEAKER_00")
        name_entry = tk.Entry(input_frame, font=("Arial", 12), width=40)
        name_entry.insert(0, speaker_id)  # Pre-fill with speaker ID
        name_entry.pack(fill=tk.X, pady=(5, 10))
        name_entry.select_range(0, tk.END)  # Select all text for easy replacement
        name_entry.focus()

        # Button frame (only Submit button now)
        button_frame = tk.Frame(input_frame, bg="#ecf0f1")
        button_frame.pack()

        submit_btn = tk.Button(
            button_frame,
            text="Submit",
            command=on_submit,
            bg="#27ae60",
            fg="white",
            font=("Arial", 11, "bold"),
            cursor="hand2",
            width=20
        )
        submit_btn.pack()

        # Bind Enter key to submit
        name_entry.bind('<Return>', lambda e: on_submit())

        # Wait for dialog to close
        dialog.wait_window()

        # Cleanup pygame
        pygame.mixer.quit()

        return result[0] if result[0] is not None else ""

    def map_speaker_names_with_samples(self, speakers: List[str], aligned_segments: List[Dict], audio_file: str = None) -> Dict[str, str]:
        """
        GUI to map speaker IDs to real names, with audio playback if available

        Args:
            speakers: List of speaker IDs (e.g., ['SPEAKER_00', 'SPEAKER_01'])
            aligned_segments: List of segments with speaker labels and text
            audio_file: Path to original audio file for extracting playable clips (optional)

        Returns:
            Dictionary mapping speaker IDs to names
        """
        # If audio file provided, use audio playback GUI
        if audio_file:
            print("\nPreparing speaker identification with audio playback...")
            import tempfile
            import os

            # Create temporary directory for audio clips
            temp_dir = tempfile.mkdtemp(prefix="ttrpg_speaker_clips_")
            audio_clips_to_delete = []

            try:
                speaker_map = {}
                speaker_data = {}  # Store quotes and clips for each identified speaker

                # Collect sample quotes and audio clips for each speaker
                for speaker_id in speakers:
                    print(f"  Extracting audio samples for {speaker_id}...")
                    quotes = []
                    audio_clips = []
                    speaker_segments = [seg for seg in aligned_segments if seg["speaker"] == speaker_id and len(seg["text"].strip()) > 20]

                    # Take quotes evenly distributed throughout their speaking time
                    if len(speaker_segments) > 10:
                        step = len(speaker_segments) // 10
                        sampled = [speaker_segments[i * step] for i in range(10)]
                    else:
                        sampled = speaker_segments

                    for idx, seg in enumerate(sampled):
                        # Format timestamp
                        hours = int(seg["start"] // 3600)
                        mins = int((seg["start"] % 3600) // 60)
                        secs = int(seg["start"] % 60)
                        if hours > 0:
                            timestamp = f"{hours}:{mins:02d}:{secs:02d}"
                        else:
                            timestamp = f"{mins}:{secs:02d}"
                        quotes.append(f"[{timestamp}] {seg['text'].strip()}")

                        # Extract audio clip
                        clip_path = os.path.join(temp_dir, f"{speaker_id}_clip_{idx}.wav")
                        self.extract_audio_clip(audio_file, seg["start"], seg["end"], clip_path)
                        audio_clips.append(clip_path)
                        audio_clips_to_delete.append(clip_path)

                    # Show GUI with audio playback (pass previous speaker data for comparison)
                    name = self.speaker_identification_dialog(speaker_id, quotes, audio_clips, speaker_data)

                    if name and name.strip():
                        assigned_name = name.strip()
                        speaker_map[speaker_id] = assigned_name
                        # Store this speaker's data for comparison with future speakers
                        speaker_data[assigned_name] = {
                            'quotes': quotes,
                            'clips': audio_clips
                        }
                    else:
                        speaker_map[speaker_id] = speaker_id
                        speaker_data[speaker_id] = {
                            'quotes': quotes,
                            'clips': audio_clips
                        }

                print(f"\nSpeaker mapping: {speaker_map}")
                return speaker_map

            finally:
                # Clean up audio clips
                print("\nCleaning up temporary audio clips...")
                for clip_file in audio_clips_to_delete:
                    try:
                        if os.path.exists(clip_file):
                            os.remove(clip_file)
                    except Exception as e:
                        print(f"  Warning: Could not delete {clip_file}: {e}")

                try:
                    os.rmdir(temp_dir)
                    print("  Temporary files cleaned up!")
                except OSError:
                    # Directory not empty or other OS error - ignore
                    pass

        # Fallback: text-only mode if no audio file
        else:
            print("\nOpening speaker naming dialog with transcript samples...")
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            speaker_map = {}

            # Collect sample quotes for each speaker
            speaker_samples = {}
            for speaker_id in speakers:
                quotes = []
                speaker_segments = [seg for seg in aligned_segments if seg["speaker"] == speaker_id and len(seg["text"].strip()) > 20]

                if len(speaker_segments) > 10:
                    step = len(speaker_segments) // 10
                    sampled = [speaker_segments[i * step] for i in range(10)]
                else:
                    sampled = speaker_segments

                for seg in sampled:
                    hours = int(seg["start"] // 3600)
                    mins = int((seg["start"] % 3600) // 60)
                    secs = int(seg["start"] % 60)
                    if hours > 0:
                        timestamp = f"{hours}:{mins:02d}:{secs:02d}"
                    else:
                        timestamp = f"{mins}:{secs:02d}"
                    quotes.append(f"[{timestamp}] {seg['text'].strip()}")

                speaker_samples[speaker_id] = quotes

            # Ask for names with context
            for speaker_id in speakers:
                samples = speaker_samples.get(speaker_id, [])

                if samples:
                    samples_text = "\n\n".join(samples)
                    prompt = (
                        f"Who is {speaker_id}?\n\n"
                        f"Sample quotes from this speaker:\n"
                        f"{'-'*50}\n"
                        f"{samples_text}\n"
                        f"{'-'*50}\n\n"
                        f"Enter a name (e.g., 'DM', 'Alice', 'Bob'):\n"
                        f"Or press Cancel to keep as '{speaker_id}'"
                    )
                else:
                    prompt = (
                        f"Who is {speaker_id}?\n\n"
                        f"(No transcript samples available)\n\n"
                        f"Enter a name (e.g., 'DM', 'Alice', 'Bob'):\n"
                        f"Or press Cancel to keep as '{speaker_id}'"
                    )

                name = simpledialog.askstring(
                    "Speaker Identification",
                    prompt,
                    parent=root
                )

                if name and name.strip():
                    speaker_map[speaker_id] = name.strip()
                else:
                    speaker_map[speaker_id] = speaker_id

            root.destroy()
            print(f"Speaker mapping: {speaker_map}")
            return speaker_map

    def get_party_members_dialog(self, parent) -> List[tuple]:
        """
        Custom dialog for adding party members with +/- buttons

        Args:
            parent: Parent tkinter window

        Returns:
            List of (character_name, role) tuples
        """
        dialog = tk.Toplevel(parent)
        dialog.title("Campaign Context - Party Members")
        dialog.geometry("500x400")
        dialog.attributes('-topmost', True)

        # Instructions
        instructions = tk.Label(
            dialog,
            text="Add party members and their classes/roles:\n"
                 "Use + to add more characters, - to remove the last one",
            justify=tk.LEFT,
            padx=10,
            pady=10
        )
        instructions.pack()

        # Scrollable frame for character entries
        canvas = tk.Canvas(dialog)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Store entry widgets
        character_entries = []

        def add_character_row():
            """Add a new character/role entry row"""
            row_frame = tk.Frame(scrollable_frame)
            row_frame.pack(fill=tk.X, padx=10, pady=5)

            # Character name
            char_label = tk.Label(row_frame, text="Character:", width=10)
            char_label.pack(side=tk.LEFT)
            char_entry = tk.Entry(row_frame, width=20)
            char_entry.pack(side=tk.LEFT, padx=5)

            # Role/class
            role_label = tk.Label(row_frame, text="Role:", width=8)
            role_label.pack(side=tk.LEFT)
            role_entry = tk.Entry(row_frame, width=20)
            role_entry.pack(side=tk.LEFT, padx=5)

            character_entries.append((row_frame, char_entry, role_entry))

            # Update canvas scroll region
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

        def remove_character_row():
            """Remove the last character/role entry row"""
            if character_entries:
                row_frame, _, _ = character_entries.pop()
                row_frame.destroy()

                # Update canvas scroll region
                canvas.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))

        # Button frame
        button_frame = tk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        # +/- buttons
        add_btn = tk.Button(button_frame, text="+", command=add_character_row, width=5, font=("Arial", 14, "bold"))
        add_btn.pack(side=tk.LEFT, padx=5)

        remove_btn = tk.Button(button_frame, text="-", command=remove_character_row, width=5, font=("Arial", 14, "bold"))
        remove_btn.pack(side=tk.LEFT, padx=5)

        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)

        # Add first character row by default
        add_character_row()

        # OK/Cancel buttons
        result = []

        def on_ok():
            """Collect all character entries"""
            for _, char_entry, role_entry in character_entries:
                char_name = char_entry.get().strip()
                role = role_entry.get().strip()
                if char_name:  # Only add if character name is provided
                    result.append((char_name, role if role else "Unknown"))
            dialog.destroy()

        def on_cancel():
            """Cancel and return empty list"""
            dialog.destroy()

        ok_cancel_frame = tk.Frame(dialog)
        ok_cancel_frame.pack(fill=tk.X, padx=10, pady=10)

        ok_btn = tk.Button(ok_cancel_frame, text="OK", command=on_ok, width=10)
        ok_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = tk.Button(ok_cancel_frame, text="Cancel", command=on_cancel, width=10)
        cancel_btn.pack(side=tk.LEFT, padx=5)

        # Wait for dialog to close
        dialog.wait_window()

        return result

    def get_campaign_context(self) -> Dict[str, str]:
        """
        Get campaign context information from user via GUI

        Returns:
            Dictionary with system, setting, and party member information
        """
        print("\n" + "="*60)
        print("CAMPAIGN CONTEXT")
        print("="*60)
        print("Provide context to help the AI better summarize your session.")
        print("This information will be shown to you for review before summarization.")
        print("="*60 + "\n")

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Get game system
        system = simpledialog.askstring(
            "Campaign Context - Game System",
            "What game system are you playing?\n\n"
            "Examples: D&D 5e, Pathfinder 2e, Call of Cthulhu, Vampire: The Masquerade\n\n"
            "Press Cancel to skip:",
            parent=root
        )

        # Get setting
        setting = simpledialog.askstring(
            "Campaign Context - Setting",
            "What is the campaign setting?\n\n"
            "Examples: Forgotten Realms, homebrew cyberpunk, 1920s Arkham, modern day vampire politics\n\n"
            "Press Cancel to skip:",
            parent=root
        )

        # Get party members with custom dialog
        party_members_list = self.get_party_members_dialog(root)

        # Format party members as comma-separated list
        party_members = ", ".join([f"{name} ({role})" for name, role in party_members_list]) if party_members_list else ""

        root.destroy()

        context = {
            "system": system.strip() if system else "",
            "setting": setting.strip() if setting else "",
            "party_members": party_members
        }

        # Display the context that will be used
        print("\n" + "="*60)
        print("CAMPAIGN CONTEXT (will be provided to AI)")
        print("="*60)
        print(f"System: {context['system'] or '(Not provided)'}")
        print(f"Setting: {context['setting'] or '(Not provided)'}")
        print(f"Party Members: {context['party_members'] or '(Not provided)'}")
        print("="*60 + "\n")

        return context

    def split_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Truncate segments when other speakers start talking.

        When faster-whisper creates long segments but conversation is taking turns,
        we truncate each segment at the point where the next different speaker starts.

        This preserves chronological conversation flow.

        Args:
            segments: List of segments sorted by start time

        Returns:
            List of segments with end times adjusted
        """
        if not segments:
            return []

        result = []

        for i, seg in enumerate(segments):
            # Copy segment
            new_seg = {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }

            # Look for the next segment from a DIFFERENT speaker
            for j in range(i + 1, len(segments)):
                next_seg = segments[j]
                if next_seg["speaker"] != seg["speaker"]:
                    # If this next different-speaker segment starts BEFORE our current segment ends,
                    # truncate our segment to end when they start speaking
                    if next_seg["start"] < new_seg["end"]:
                        new_seg["end"] = next_seg["start"]
                    break  # Only look at the immediate next different speaker

            result.append(new_seg)

        return result

    def merge_consecutive_speaker_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge segments from the same speaker that are truly consecutive.

        Only merges when:
        - Same speaker
        - No other speaker spoke in between (in the sorted timeline)

        This preserves natural back-and-forth conversation flow.

        Args:
            segments: List of segments with 'speaker', 'start', 'end', 'text'
                     (MUST be pre-sorted by start time)

        Returns:
            List of merged segments
        """
        if not segments:
            return []

        merged = []
        current = {
            "speaker": segments[0].get("speaker", "UNKNOWN"),
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "text": segments[0]["text"].strip()
        }

        for seg in segments[1:]:
            speaker = seg.get("speaker", "UNKNOWN")

            # Only merge if it's the IMMEDIATE next segment from same speaker
            # If speaker changed and changed back, don't merge (preserve conversation flow)
            if speaker == current["speaker"]:
                # Same speaker continuing - merge this segment
                current["end"] = seg["end"]
                current["text"] += " " + seg["text"].strip()
            else:
                # Different speaker - save current and start new
                merged.append(current)
                current = {
                    "speaker": speaker,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                }

        # Don't forget the last segment
        merged.append(current)

        return merged

    def chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS, overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS) -> List[str]:
        """
        Split text into overlapping chunks for processing

        Args:
            text: Full text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap  # Move back by overlap amount

        print(f"Created {len(chunks)} chunks from text")
        return chunks

    def create_short_context(self, detailed_summary: str) -> str:
        """
        Create a very brief summary of a chunk summary for progressive context

        Args:
            detailed_summary: The detailed chunk summary

        Returns:
            Short 2-3 sentence context summary
        """
        prompt = f"""Summarize the following TTRPG session segment in 2-3 sentences (50 words max).
Focus ONLY on:
- Key events and plot developments
- Important NPC names and locations
- Critical decisions or discoveries

Do NOT include: combat details, specific dialogue, minor encounters.

Full summary:
{detailed_summary}

Brief summary (2-3 sentences):"""

        response = ollama.chat(
            model=self.ollama_model,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'num_gpu': 99,
                'num_thread': 8,
                'num_ctx': 4096,  # Small context for this quick task
                'num_predict': 128,  # Very short output (50 words)
                'temperature': 0.3,  # Low temperature for factual summary
            },
            keep_alive='5m'  # Keep model loaded for 5 minutes to avoid reloading between chunks
        )

        return response['message']['content'].strip()

    def summarize_chunk(self, chunk: str, chunk_number: int, total_chunks: int = None, context: Dict[str, str] = None, previous_context: str = "") -> str:
        """
        Summarize a single chunk using Ollama with progressive context from previous chunks

        Args:
            chunk: Text chunk to summarize
            chunk_number: Index of this chunk
            total_chunks: Total number of chunks for progress tracking
            context: Campaign context (system, setting, party members)
            previous_context: Brief summary of what happened in previous chunks

        Returns:
            Summary of the chunk
        """
        context = context or {}
        progress = f" ({chunk_number}/{total_chunks})" if total_chunks else ""
        print(f"\nSummarizing chunk {chunk_number}{progress}...")

        # Build context header
        context_header = ""
        if context.get("system"):
            context_header += f"Game System: {context['system']}\n"
        if context.get("setting"):
            context_header += f"Campaign Setting: {context['setting']}\n"
        if context.get("party_members"):
            context_header += f"Party Members: {context['party_members']}\n"

        # Note: Character context is NOT included in chunk summaries to reduce noise
        # Characters are tracked and will be available in the final summary if needed

        if context_header:
            context_header = f"\n**CAMPAIGN CONTEXT:**\n{context_header}\n"

        # Build previous session context
        previous_context_section = ""
        if previous_context:
            previous_context_section = f"\n**WHAT HAPPENED EARLIER THIS SESSION:**\n{previous_context}\n"

        # Determine session position for context
        if total_chunks == 1:
            # Single chunk = entire session
            session_position = f"You are summarizing a complete TTRPG session.\nThis is the FULL session from beginning to end - include how it started, what happened, and how it concluded."
        elif chunk_number == 1 and total_chunks > 1:
            # First chunk of multi-part
            session_position = f"You are summarizing the BEGINNING of a TTRPG session (part {chunk_number} of {total_chunks}).\nNote how the session opened and what the initial situation was."
        elif chunk_number == total_chunks:
            # Last chunk of multi-part
            session_position = f"You are summarizing the END of a TTRPG session (part {chunk_number} of {total_chunks}).\nThis is the final part - note how the session concluded and where it left off."
        else:
            # Middle chunk
            session_position = f"You are summarizing part {chunk_number} of {total_chunks} from an ongoing TTRPG session.\nThis is a CONTINUATION of the session, not the start or end."

        prompt = f"""{session_position}
{context_header}{previous_context_section}
CRITICAL INSTRUCTIONS:
1. ONLY summarize what is ACTUALLY in the transcript below
2. DO NOT invent, fabricate, or make up ANY information
3. If the transcript doesn't appear to be a TTRPG session (e.g., it's a podcast, conversation, or other content), say: "WARNING: This does not appear to be TTRPG content. The transcript contains: [brief description of actual content]"
4. DO NOT create fantasy stories, character names, locations, or events that aren't explicitly mentioned
5. If something is unclear or ambiguous, note it as such - don't guess

Write a NARRATIVE summary (aim for {self.chunk_target_words} words) that tells the story of what happened in this part of the session.

STYLE GUIDELINES:
- Write in a flowing narrative style, like recounting the session to someone who wasn't there
- Focus on the story and what the characters DID, not on categorizing information
- Naturally weave in details about NPCs, items, and decisions as they come up in the story
- Use character names and be specific about locations and events
- Avoid section headers and bullet points - write in paragraphs that flow together
- Start with what was happening and build from there chronologically

WHAT TO INCLUDE:
- The narrative progression: what happened, where the party went, what they encountered
- Character actions and decisions: what did each PC do and why?
- NPCs and dialogue: who did they meet and what was said?
- Combat and challenges: describe encounters as part of the story flow
- Items and discoveries: mention what was found naturally as it happens in the narrative
- Consequences and outcomes: what resulted from the party's actions?

Think of this as writing a session recap that captures the experience and helps players remember what happened, not as creating a structured database of information.

Transcript chunk:
{chunk}

Detailed Summary (ONLY based on actual transcript content):"""

        # Use streaming to show progress
        import time
        start_time = time.time()
        summary = ""
        token_count = 0
        last_update = time.time()

        # Check if context size would exceed VRAM and auto-convert to unlimited if needed
        safe_chunk_context = get_safe_context_size(self.ollama_model, self.chunk_context_size)

        # Build options for ollama
        chunk_options = {
            'num_gpu': 99,  # Use all available GPU layers
            'num_thread': 8,  # Parallel processing threads
            'num_predict': 1024,  # Limit output to 1024 tokens (~750 words)
        }

        # Add context size if specified (None = unlimited)
        if safe_chunk_context is not None:
            chunk_options['num_ctx'] = safe_chunk_context

        # Retry logic for Ollama memory errors
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Stream the response
                stream = ollama.chat(
                    model=self.ollama_model,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }],
                    options=chunk_options,
                    keep_alive='5m',  # Keep model loaded for 5 minutes to avoid reloading between chunks
                    stream=True  # Enable streaming
                )

                # Process stream and show progress
                for chunk_data in stream:
                    if 'message' in chunk_data and 'content' in chunk_data['message']:
                        content = chunk_data['message']['content']
                        summary += content
                        token_count += 1

                        # Update progress every second
                        current_time = time.time()
                        if current_time - last_update >= 1.0:
                            elapsed = current_time - start_time
                            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

                            # Estimate progress (assume ~1024 tokens max for chunks)
                            progress_pct = min(100, (token_count / 1024) * 100)

                            # Compact progress bar for chunks
                            bar_length = 20
                            filled = int(bar_length * progress_pct / 100)
                            bar = '█' * filled + '░' * (bar_length - filled)

                            print(f"\r  [{bar}] {progress_pct:.0f}% | {token_count} tok | {tokens_per_sec:.1f} tok/s   ",
                                  end='', flush=True)

                            last_update = current_time

                # Success! Break out of retry loop
                break

            except Exception as e:
                error_msg = str(e)
                retry_count += 1

                # Check if it's a memory/VRAM error
                if "llama runner process has terminated" in error_msg or "GGML_ASSERT" in error_msg or "status code: 500" in error_msg:
                    print(f"\n  ⚠️  Ollama memory error on chunk {chunk_number} (attempt {retry_count}/{max_retries})")

                    if retry_count < max_retries:
                        print(f"  🔄 Unloading model and retrying in 5 seconds...")

                        # Unload the model to free memory
                        try:
                            import subprocess
                            subprocess.run(['ollama', 'stop', self.ollama_model],
                                         capture_output=True, timeout=10)
                        except:
                            pass

                        # Wait for memory to clear
                        time.sleep(5)

                        # Reset for retry
                        summary = ""
                        token_count = 0
                        start_time = time.time()
                        last_update = time.time()
                    else:
                        print(f"  ❌ Failed after {max_retries} attempts. Skipping this chunk.")
                        summary = f"[ERROR: Failed to summarize chunk {chunk_number} due to memory issues after {max_retries} attempts]"
                        break
                else:
                    # Different error, don't retry
                    print(f"\n  ❌ Error summarizing chunk {chunk_number}: {error_msg}")
                    raise

        # Final update
        elapsed = time.time() - start_time
        print(f"\r  ✓ Chunk {chunk_number} complete: {len(summary)} chars, {token_count} tokens in {int(elapsed)}s" + " "*20)

        # Brief pause so users can see the completion message
        time.sleep(0.5)

        # Extract characters/NPCs from this chunk
        self.extract_characters_from_summary(summary, chunk_number, self.current_session_name)

        # Debug: Show current character count after extraction
        if self.characters:
            print(f"  [Debug] Total characters in memory: {len(self.characters)}")

        return summary

    def extract_characters_from_summary(self, summary: str, chunk_number: int, session_name: str = None):
        """
        Extract character/NPC names and information from a summary using Ollama

        Args:
            summary: The chunk summary text
            chunk_number: Which chunk this is from
            session_name: Name of the session (e.g., "Episode 1") for tracking first appearance
        """
        try:
            # Use a quick extraction prompt
            extraction_prompt = f"""Extract ALL character and NPC names from this TTRPG session summary.

IMPORTANT RULES FOR NAMES:
- Use ONLY the character's core name (e.g., "Jasper", "Annabelle", "Baron Abrams")
- DO NOT add qualifiers like "(Player: Name)", "(NPC)", "(mentioned)", "(via remote)", etc.
- DO NOT add parenthetical information to names
- If a character has multiple forms of their name, use the most common/complete one
- Examples: Use "Victor Temple" not "Victor Temple (Player: Dave)" or "Victor Temple (via remote support)"

For EACH character/NPC mentioned, provide:
1. Their core name only (or role if unnamed, like "Tavern Keeper", "Mysterious Figure")
2. A brief 1-sentence description of who they are or what they did

Format your response EXACTLY as:
NAME: [character name - NO parentheses or qualifiers]
DESCRIPTION: [one sentence description]

NAME: [next character]
DESCRIPTION: [one sentence description]

If NO characters or NPCs are mentioned, respond with: "NONE"

Summary to analyze:
{summary}"""

            # Call Ollama with minimal tokens (we just need a list)
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': extraction_prompt}],
                options={
                    'num_gpu': 99,
                    'num_thread': 4,
                    'num_ctx': 2048,  # Small context, just for extraction
                    'num_predict': 512,  # Limit response length
                    'temperature': 0.3,  # Low temp for factual extraction
                },
                stream=False
            )

            extraction = response['message']['content']

            # Parse the extraction
            if "NONE" in extraction.upper() and len(extraction) < 50:
                return  # No characters found

            # Extract NAME/DESCRIPTION pairs
            import re
            name_pattern = r'NAME:\s*(.+?)(?=\n|$)'
            desc_pattern = r'DESCRIPTION:\s*(.+?)(?=\n|$)'

            names = re.findall(name_pattern, extraction, re.IGNORECASE)
            descriptions = re.findall(desc_pattern, extraction, re.IGNORECASE)

            # Match names with descriptions
            new_characters = []
            for i, name in enumerate(names):
                name = name.strip()
                desc = descriptions[i].strip() if i < len(descriptions) else "No description"

                # Normalize name by removing common qualifiers/parentheticals
                # e.g., "Jasper (Player: Alexander)" -> "Jasper"
                normalized_name = name
                import re
                # Remove patterns like "(Player: X)", "(mentioned...)", "(Potential NPC)", etc.
                normalized_name = re.sub(r'\s*\([^)]*\)\s*$', '', normalized_name).strip()

                # Also remove "via X" suffixes
                normalized_name = re.sub(r'\s+via\s+.*$', '', normalized_name, flags=re.IGNORECASE).strip()

                # Add or update character (use normalized name as key)
                if normalized_name in self.characters:
                    self.characters[normalized_name]['mentions'] += 1

                    # Merge information: Check if new description adds unique information
                    old_desc = self.characters[normalized_name]['description'].lower()
                    new_desc_lower = desc.lower()

                    # If descriptions are substantially different, merge them
                    # (avoids exact duplicates but preserves new info)
                    if new_desc_lower not in old_desc and old_desc not in new_desc_lower:
                        # New information found - append it
                        self.characters[normalized_name]['description'] = f"{self.characters[normalized_name]['description']}; {desc}"
                    elif len(desc) > len(self.characters[normalized_name]['description']):
                        # New description is more detailed, replace it
                        self.characters[normalized_name]['description'] = desc
                    # Otherwise keep existing description (new one is redundant/shorter)
                else:
                    # Use session name if available, otherwise fall back to chunk number
                    first_appearance = f"{session_name} (Chunk {chunk_number})" if session_name else f"Chunk {chunk_number}"
                    self.characters[normalized_name] = {
                        'first_appearance': first_appearance,
                        'description': desc,
                        'mentions': 1
                    }
                    new_characters.append(normalized_name)

            # Show new characters discovered in this chunk
            if new_characters:
                print(f"  📝 New characters/NPCs: {', '.join(new_characters)}")

        except Exception as e:
            # Log extraction failures but don't crash
            print(f"  ⚠️  Character extraction failed for chunk {chunk_number}: {e}")

    def load_campaign_characters(self):
        """Load characters from the campaign character file"""
        try:
            print(f"📂 Loading campaign characters from: {self.campaign_character_file}")
            with open(self.campaign_character_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse character entries from the file
            # Format: **Name**\n  First appeared: [session info]\n  Total mentions: X\n  Description: ...
            import re

            # Find all character blocks (flexible pattern to match session names)
            char_pattern = r'\*\*(.+?)\*\*\s*\n\s*First appeared:\s*(.+?)\s*\n\s*Total mentions:\s*(\d+)\s*\n\s*Description:\s*(.+?)(?=\n\n|\n\*\*|$)'
            matches = re.findall(char_pattern, content, re.DOTALL | re.IGNORECASE)

            for name, first_appearance, mentions, description in matches:
                name = name.strip()
                self.characters[name] = {
                    'first_appearance': first_appearance.strip(),
                    'description': description.strip(),
                    'mentions': int(mentions)
                }

            if self.characters:
                print(f"  ✓ Loaded {len(self.characters)} existing character(s) from campaign file")
                print(f"  Characters: {', '.join(list(self.characters.keys())[:5])}" +
                      (f" + {len(self.characters)-5} more" if len(self.characters) > 5 else ""))
            else:
                print(f"  Campaign file exists but no characters found (new campaign)")

        except Exception as e:
            print(f"⚠️  Could not load campaign characters: {e}")
            print(f"  Starting with empty character list")

    def save_campaign_characters(self):
        """Save/update the campaign character file with all tracked characters"""
        if not self.campaign_character_file:
            print("⚠️  No campaign character file specified - skipping save")
            return  # No campaign file specified

        print(f"\n📝 Saving {len(self.characters)} characters to campaign file...")
        print(f"   File: {self.campaign_character_file}")

        try:
            # Read existing content to preserve campaign header
            campaign_name = "Campaign"
            if Path(self.campaign_character_file).exists():
                with open(self.campaign_character_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith("CAMPAIGN:"):
                            campaign_name = line.replace("CAMPAIGN:", "").strip()
                            break

            # Write updated character list
            with open(self.campaign_character_file, 'w', encoding='utf-8') as f:
                f.write(f"{BANNER_SINGLE}\n")
                f.write(f"CAMPAIGN: {campaign_name}\n")
                f.write(f"{BANNER_SINGLE}\n\n")

                if not self.characters:
                    f.write("No characters tracked yet.\n\n")
                    print("   ⚠️  Character dictionary is empty!")
                else:
                    f.write(f"Total Characters/NPCs Tracked: {len(self.characters)}\n\n")
                    # Sort by first appearance
                    sorted_chars = sorted(self.characters.items(),
                                        key=lambda x: x[1]['first_appearance'])

                    for name, info in sorted_chars:
                        f.write(f"**{name}**\n")
                        f.write(f"  First appeared: {info['first_appearance']}\n")
                        f.write(f"  Total mentions: {info['mentions']}\n")
                        f.write(f"  Description: {info['description']}\n\n")

            print(f"✓ Campaign character file updated: {self.campaign_character_file}")
            print(f"  Total characters tracked: {len(self.characters)}")

        except Exception as e:
            print(f"⚠️  Could not save campaign characters: {e}")

    def show_final_summary_config(self, chunk_summaries: List[str], default_target_words: int = 1200, default_context_size: int = None) -> tuple:
        """
        Show a dialog to configure final summary settings based on actual chunk data

        Args:
            chunk_summaries: List of chunk summaries to analyze
            default_target_words: Default target word count
            default_context_size: Default context size (None = unlimited)

        Returns:
            Tuple of (model, target_words, context_size, include_characters) or None if cancelled
        """
        import tkinter as tk

        # Calculate token estimate with 20% safety buffer
        combined = "\n\n".join([f"Part {i+1}:\n{summary}" for i, summary in enumerate(chunk_summaries)])
        total_chars = len(combined)
        base_tokens = total_chars // 4
        # Add 20% buffer for system prompts, formatting, and tokenizer variance
        estimated_input_tokens = int(base_tokens * 1.2)
        word_count = len(combined.split())

        # Calculate recommended minimum (rounds up to nearest 2048)
        recommended = ((estimated_input_tokens // 2048) + 1) * 2048

        # Get VRAM info and set smart default
        total_vram, available_vram, gpu_name = get_available_vram()

        # Context size limits based on VRAM (rough estimates for Qwen 14B)
        # Use slightly lower thresholds to account for actual available VRAM
        # 6GB+ VRAM ≈ 8192 tokens, 10GB+ ≈ 16384, 14GB+ ≈ 32768, 20GB+ ≈ 65536+
        vram_to_context = {
            6: 8192,
            10: 16384,
            14: 32768,
            20: 65536,
            28: 131072
        }

        max_safe_context = 32768  # Default safe value
        # Use available VRAM (not total) since other models may already be loaded
        if available_vram > 0:
            for vram_gb, context_tokens in sorted(vram_to_context.items()):
                if available_vram >= vram_gb:
                    max_safe_context = context_tokens

        # Set smart default: use recommended size, but cap at VRAM limits
        if default_context_size is not None:
            # User explicitly set a value, use it
            smart_default = default_context_size
        elif recommended <= max_safe_context:
            # Recommended size fits in VRAM, use it
            smart_default = recommended
        else:
            # Recommended exceeds VRAM capacity, cap at max safe value
            smart_default = max_safe_context

        result = [None]  # Store result

        def on_ok():
            result[0] = (model_var.get(), target_words_var.get(), context_var.get(), include_chars_var.get())
            dialog.destroy()

        def on_cancel():
            result[0] = None
            dialog.destroy()

        # Create dialog
        dialog = tk.Toplevel()
        dialog.title("Final Summary Configuration")
        dialog.geometry("650x680")
        dialog.configure(bg="#ecf0f1")
        dialog.attributes('-topmost', True)
        dialog.grab_set()

        # Header
        header_frame = tk.Frame(dialog, bg="#3498db", padx=20, pady=15)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="Configure Final Summary",
                font=("Arial", 16, "bold"), bg="#3498db", fg="white").pack()
        tk.Label(header_frame, text="Review the analysis and adjust settings before generating the final summary",
                font=("Arial", 10), bg="#3498db", fg="#ecf0f1").pack()

        # Content frame
        content_frame = tk.Frame(dialog, bg="white", padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Analysis section
        analysis_frame = tk.LabelFrame(content_frame, text="📊 Input Analysis",
                                      bg="white", font=("Arial", 11, "bold"), padx=15, pady=15)
        analysis_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(analysis_frame, text=f"Total chunk summaries: {len(chunk_summaries)}",
                bg="white", font=("Arial", 10), anchor=tk.W).pack(fill=tk.X, pady=2)
        tk.Label(analysis_frame, text=f"Combined length: {word_count:,} words",
                bg="white", font=("Arial", 10), anchor=tk.W).pack(fill=tk.X, pady=2)
        tk.Label(analysis_frame, text=f"Estimated tokens: ~{estimated_input_tokens:,} tokens",
                bg="white", font=("Arial", 10, "bold"), fg="#2c3e50", anchor=tk.W).pack(fill=tk.X, pady=2)
        tk.Label(analysis_frame, text=f"Recommended minimum context: {recommended:,} tokens",
                bg="white", font=("Arial", 10), fg="#27ae60", anchor=tk.W).pack(fill=tk.X, pady=2)

        # VRAM info
        if total_vram > 0:
            vram_text = f"GPU: {gpu_name} | Total: {total_vram:.1f} GB | Available: {available_vram:.1f} GB"
            vram_color = "#3498db"
        else:
            vram_text = f"GPU: {gpu_name} (CPU mode)"
            vram_color = "#95a5a6"
        tk.Label(analysis_frame, text=vram_text,
                bg="white", font=("Arial", 9), fg=vram_color, anchor=tk.W).pack(fill=tk.X, pady=2)

        # Settings section
        settings_frame = tk.LabelFrame(content_frame, text="⚙️ Summary Settings",
                                      bg="white", font=("Arial", 11, "bold"), padx=15, pady=15)
        settings_frame.pack(fill=tk.X, pady=(0, 15))

        # Get available Ollama models
        from tkinter import ttk
        ollama_models = get_ollama_models()
        model_names = [m[0] for m in ollama_models] if ollama_models else ["qwen2.5:14b"]

        # Model selection
        model_var = tk.StringVar(value=self.ollama_final_model)
        model_frame = tk.Frame(settings_frame, bg="white")
        model_frame.pack(fill=tk.X, pady=5)
        tk.Label(model_frame, text="Model:", bg="white", font=("Arial", 10), width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Combobox(model_frame, textvariable=model_var, values=model_names,
                    state="readonly", width=25, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        # Target words
        target_words_var = tk.IntVar(value=default_target_words)
        words_frame = tk.Frame(settings_frame, bg="white")
        words_frame.pack(fill=tk.X, pady=5)
        tk.Label(words_frame, text="Target words:", bg="white", font=("Arial", 10), width=15, anchor=tk.W).pack(side=tk.LEFT)
        tk.Spinbox(words_frame, from_=500, to=5000, increment=100, textvariable=target_words_var,
                  width=10, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        # Character inclusion checkbox
        include_chars_var = tk.BooleanVar(value=True)  # Default to True for backward compatibility
        chars_frame = tk.Frame(settings_frame, bg="white")
        chars_frame.pack(fill=tk.X, pady=5)
        tk.Label(chars_frame, text="Include characters:", bg="white", font=("Arial", 10), width=15, anchor=tk.W).pack(side=tk.LEFT)
        char_count = len(self.characters) if self.characters else 0
        tk.Checkbutton(chars_frame, text=f"Include character list in summary ({char_count} character{'s' if char_count != 1 else ''} tracked)",
                      variable=include_chars_var, bg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)

        # Context size
        context_var = tk.IntVar(value=smart_default)
        context_frame = tk.Frame(settings_frame, bg="white")
        context_frame.pack(fill=tk.X, pady=5)
        tk.Label(context_frame, text="Context size:", bg="white", font=("Arial", 10), width=15, anchor=tk.W).pack(side=tk.LEFT)

        context_entry_var = tk.StringVar()

        def update_context_display(*_):
            val = context_var.get()
            if val >= 999999:
                context_entry_var.set("Unlimited")
            else:
                context_entry_var.set(f"{val:,}")

        def increment_context():
            current = context_var.get()
            if current >= 999999:
                # Already at unlimited, stay there
                pass
            elif current >= 131072:
                # Jump to unlimited from 131K
                context_var.set(999999)
            else:
                # Increment by 2048
                context_var.set(current + 2048)
            update_context_display()

        def decrement_context():
            current = context_var.get()
            if current >= 999999:
                # From unlimited, go back to 131K
                context_var.set(131072)
            else:
                # Decrement by 2048, minimum 2048
                context_var.set(max(current - 2048, 2048))
            update_context_display()

        context_var.trace_add('write', update_context_display)
        update_context_display()

        def set_recommended():
            # Set to actual recommended value (not capped) - user will see warning if exceeds VRAM
            context_var.set(recommended)
            update_context_display()

        def set_unlimited():
            context_var.set(999999)
            update_context_display()

        tk.Entry(context_frame, textvariable=context_entry_var, width=12,
                font=("Arial", 10), state='readonly').pack(side=tk.LEFT, padx=5)
        tk.Button(context_frame, text="▲", command=increment_context, width=2).pack(side=tk.LEFT)
        tk.Button(context_frame, text="▼", command=decrement_context, width=2).pack(side=tk.LEFT, padx=(2, 0))
        tk.Button(context_frame, text="Recommended", command=set_recommended,
                 bg="#3498db", fg="white", font=("Arial", 9, "bold"),
                 padx=8, pady=2).pack(side=tk.LEFT, padx=(10, 0))
        tk.Button(context_frame, text="Unlimited", command=set_unlimited,
                 bg="#f39c12", fg="white", font=("Arial", 9, "bold"),
                 padx=8, pady=2).pack(side=tk.LEFT, padx=(5, 0))

        # VRAM estimate label
        vram_estimate_label = tk.Label(settings_frame, text="", bg="white", font=("Arial", 9))
        vram_estimate_label.pack(fill=tk.X, pady=(5, 0))

        # Warning/recommendation label
        warning_label = tk.Label(settings_frame, text="", bg="white", wraplength=500,
                                justify=tk.LEFT, font=("Arial", 9))
        warning_label.pack(fill=tk.X, pady=(10, 0))

        def update_warning(*_):
            ctx = context_var.get()
            selected_model = model_var.get()

            # Check VRAM usage
            vram_estimate = estimate_vram_usage(selected_model, ctx if ctx < 999999 else 131072)
            try:
                vram_str = vram_estimate.replace("~", "").replace("GB", "").strip()
                estimated_vram_gb = float(vram_str)
                _, available_vram_gb, _ = get_available_vram()

                # Update VRAM estimate label
                if available_vram_gb > 0 and estimated_vram_gb > available_vram_gb:
                    vram_estimate_label.config(text=f"Estimated VRAM: {vram_estimate}", fg="#e74c3c", font=("Arial", 9, "bold"))
                else:
                    vram_estimate_label.config(text=f"Estimated VRAM: {vram_estimate}", fg="#7f8c8d", font=("Arial", 9))

                # Priority 1: VRAM exceeded
                if available_vram_gb > 0 and estimated_vram_gb > available_vram_gb:
                    warning_label.config(text=f"⚠️ VRAM EXCEEDED! Estimated: {vram_estimate}, Available: {available_vram_gb:.1f} GB\nWill use shared GPU memory - expect significantly slower performance!",
                                       fg="#e74c3c", font=("Arial", 9, "bold"))
                # Priority 2: Unlimited context warning
                elif ctx >= 999999:
                    warning_label.config(text="⚠️ Unlimited may be VERY slow for long sessions!",
                                       fg="#e74c3c", font=("Arial", 9, "bold"))
                # Priority 3: Context too small (only if below recommended AND have VRAM headroom)
                elif ctx < recommended and estimated_vram_gb <= available_vram_gb:
                    # Context is below recommended AND we have VRAM available to increase it
                    warning_label.config(text=f"⚠️ Context may be too small! Some summaries may be truncated.\nRecommended: {recommended:,} tokens or more",
                                       fg="#e74c3c", font=("Arial", 9, "bold"))
                # Priority 4: Good context size
                elif ctx >= recommended:
                    warning_label.config(text="✓ Context size is adequate for all chunk summaries",
                                       fg="#27ae60", font=("Arial", 9, "bold"))
                # Priority 5: Available output tokens
                else:
                    available = ctx - estimated_input_tokens
                    warning_label.config(text=f"Context available for output: ~{available:,} tokens",
                                       fg="#7f8c8d", font=("Arial", 9))
            except Exception as e:
                # Fallback to original logic if VRAM check fails
                print(f"Warning: VRAM estimation failed: {e}")
                vram_estimate_label.config(text=f"Estimated VRAM: {vram_estimate}", fg="#7f8c8d", font=("Arial", 9))
                if ctx >= UNLIMITED_CONTEXT:
                    warning_label.config(text="⚠️ Unlimited may be VERY slow for long sessions!",
                                       fg="#e74c3c", font=("Arial", 9, "bold"))
                elif estimated_input_tokens > ctx * 0.8:
                    warning_label.config(text=f"⚠️ Context may be too small! Some summaries may be truncated.\nRecommended: {recommended:,} tokens or more",
                                       fg="#e74c3c", font=("Arial", 9, "bold"))
                elif ctx >= recommended:
                    warning_label.config(text="✓ Context size is adequate for all chunk summaries",
                                       fg="#27ae60", font=("Arial", 9, "bold"))
                else:
                    available = ctx - estimated_input_tokens
                    warning_label.config(text=f"Context available for output: ~{available:,} tokens",
                                       fg="#7f8c8d", font=("Arial", 9))

        context_var.trace_add('write', update_warning)
        model_var.trace_add('write', update_warning)
        update_warning()

        # Buttons
        button_frame = tk.Frame(dialog, bg="#ecf0f1", pady=15, padx=20)
        button_frame.pack(fill=tk.X)

        tk.Button(button_frame, text="✓ Generate Summary", command=on_ok,
                 bg="#27ae60", fg="white", font=("Arial", 12, "bold"),
                 padx=30, pady=10, cursor="hand2").pack(side=tk.RIGHT, padx=(10, 0))
        tk.Button(button_frame, text="Cancel", command=on_cancel,
                 bg="#95a5a6", fg="white", font=("Arial", 10),
                 padx=20, pady=10, cursor="hand2").pack(side=tk.RIGHT, padx=(0, 10))

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        dialog.wait_window()
        return result[0]

    def create_overall_summary(self, chunk_summaries: List[str], context: Dict[str, str] = None, final_context_size: int = None, output_file: str = None, include_characters: bool = True) -> str:
        """
        Create a final summary from all chunk summaries

        Args:
            chunk_summaries: List of summaries from each chunk
            context: Campaign context (system, setting, party members)
            final_context_size: Context window size for final summary (None = unlimited)
            output_file: Optional file path to stream summary directly to disk (reduces memory usage)
            include_characters: Whether to include character list in the summary context

        Returns:
            Overall session summary (or file path if streaming to file)
        """
        context = context or {}
        print("Creating overall session summary...")

        combined = "\n\n".join([
            f"Part {i+1}:\n{summary}"
            for i, summary in enumerate(chunk_summaries)
        ])

        # Estimate token count for the input with 20% safety buffer
        # Base estimation: chars / 4, then add buffer for system prompts and formatting
        total_chars = len(combined)
        base_tokens = total_chars // 4
        estimated_input_tokens = int(base_tokens * 1.2)  # 20% buffer
        word_count = len(combined.split())

        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY INPUT ANALYSIS")
        print(f"{'='*60}")
        print(f"Total chunk summaries: {len(chunk_summaries)}")
        print(f"Combined summary length: {word_count:,} words")
        print(f"Estimated input tokens: ~{estimated_input_tokens:,} tokens")

        # Provide context size recommendation
        if final_context_size and final_context_size < 999999:
            if estimated_input_tokens > final_context_size * 0.8:  # Using >80% of context
                print(f"⚠️  WARNING: Your context size ({final_context_size:,} tokens) may be too small!")
                recommended = ((estimated_input_tokens // 2048) + 1) * 2048  # Round up to nearest 2048
                print(f"   Recommended minimum: {recommended:,} tokens")
                print(f"   Some chunk summaries may be truncated!")
            else:
                available = final_context_size - estimated_input_tokens
                print(f"✓ Context size ({final_context_size:,} tokens) is adequate")
                print(f"  Available for output: ~{available:,} tokens")
        else:
            print(f"✓ Using unlimited context - all summaries will be included")
        print(f"{'='*60}\n")

        # Build context header
        context_header = ""
        if context.get("system"):
            context_header += f"Game System: {context['system']}\n"
        if context.get("setting"):
            context_header += f"Campaign Setting: {context['setting']}\n"
        if context.get("party_members"):
            context_header += f"Party Members: {context['party_members']}\n"

        # Add all known characters for final summary if requested
        if include_characters and self.characters:
            # Show all characters sorted by mention count
            sorted_chars = sorted(self.characters.items(), key=lambda x: x[1]['mentions'], reverse=True)
            context_header += f"\n**CHARACTERS ENCOUNTERED THIS SESSION:**\n"
            context_header += "Use this reference to maintain consistency when writing the final summary.\n"
            context_header += "⚠️ ONLY include these in your summary if they appear in the part summaries below!\n\n"
            for name, info in sorted_chars:
                context_header += f"- {name}: {info['description']} (appeared in {info['mentions']} part(s))\n"

        if context_header:
            context_header = f"\n**CAMPAIGN CONTEXT:**\n{context_header}\n"

        prompt = f"""You are creating a comprehensive final summary of a TTRPG session.
Below are detailed summaries of different parts of the session in chronological order.
{context_header}
CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. ONLY summarize what is ACTUALLY in the part summaries below
2. DO NOT invent, fabricate, or make up ANY information not present in the summaries
3. If the summaries contain warnings that this is NOT TTRPG content (e.g., podcasts, conversations), STOP and output: "ERROR: The provided content does not appear to be from a TTRPG session. Cannot generate fantasy summary for non-TTRPG content. Actual content type: [description from warnings]"
4. DO NOT create fantasy stories, locations, NPCs, items, or events that aren't mentioned in the summaries
5. If information is missing or unclear in the summaries, acknowledge the gap - don't fill it with fiction
6. If the content IS a legitimate TTRPG session, then: This summary should be AT LEAST {self.final_summary_target_words} words. Do NOT provide a short summary. Players need detailed information to remember what happened. Be thorough and comprehensive.

Write a detailed narrative-style summary organized into these sections:

## Session Overview (100-150 words)
- Opening context: Where the party was and what they were doing
- Primary objectives or goals for this session
- Brief overview of major accomplishments

## Detailed Session Narrative (800-1000 words)
Write a chronological story of what happened. Include:
- Specific locations visited with descriptions
- NPC names and their roles (the DM speaks AS various NPCs - identify them!)
- Dialogue highlights and important conversations
- Each party member's actions and contributions
- Combat encounters with tactics and outcomes
- Skill checks, puzzles, or challenges
- Items found, rewards gained, or resources used
- Key decisions made and their immediate consequences
- Mood, tone, and roleplay moments

## Character Highlights (150-200 words)
For each party member:
- Notable actions or heroic moments
- Character development or backstory reveals
- Relationships and interactions with NPCs or other PCs
- Any new abilities, items, or character changes

## Important NPCs & Discoveries (100-150 words)
- List NPCs encountered with brief descriptions
- New information learned about the world, plot, or mysteries
- Alliances formed or broken
- Threats identified

## Session Conclusion & Next Steps (50-100 words)
- Where the session ended (location and situation)
- Immediate goals for next session
- Unresolved questions or cliffhangers
- Any preparation needed

Remember: BE DETAILED. This is a 3-hour session - {self.final_summary_target_words}+ words is appropriate to capture everything that happened.

Part summaries:
{combined}

COMPREHENSIVE SESSION SUMMARY:"""

        # Use streaming to show progress
        print("\nGenerating summary (this may take 15-45 minutes for large sessions)...")
        print("Progress: ", end='', flush=True)

        import time
        start_time = time.time()
        overall_summary = ""
        token_count = 0
        last_update = time.time()

        # Build options for ollama
        ollama_options = {
            'num_gpu': 99,  # Use all available GPU layers
            'num_thread': 8,  # Parallel processing threads
            'num_predict': 2048,  # Allow up to 2048 tokens (~1500 words) for detailed summary
            'temperature': 0.7,  # Slight creativity for narrative style
        }

        # Check if context size would exceed VRAM and auto-convert to unlimited if needed
        safe_context_size = get_safe_context_size(self.ollama_final_model, final_context_size or 999999)

        # Add context size if specified (None = unlimited)
        if safe_context_size is not None:
            ollama_options['num_ctx'] = safe_context_size
            print(f"Using context window: {safe_context_size:,} tokens")
        else:
            print("Using unlimited context window")

        # Stream the response using the final summary model
        stream = ollama.chat(
            model=self.ollama_final_model,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options=ollama_options,
            keep_alive='5m',  # Keep model loaded to avoid unloading after completion
            stream=True  # Enable streaming
        )

        # Process stream and show progress
        # If output_file specified, stream directly to file (reduces memory usage)
        if output_file:
            print(f"Streaming final summary directly to file: {output_file}")
            print("(This reduces memory usage for very long summaries)")

            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for chunk in stream:
                        # Check for stop/pause before processing each chunk
                        self.check_control_events()

                        if 'message' in chunk and 'content' in chunk['message']:
                            content = chunk['message']['content']
                            f.write(content)  # Write directly to file
                            token_count += 1

                            # Flush periodically (every 100 tokens instead of every token)
                            if token_count % 100 == 0:
                                f.flush()

                            # Update progress every 5 seconds
                            current_time = time.time()
                            if current_time - last_update >= 5.0:
                                elapsed = current_time - start_time
                                tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                                print(f"Progress: {token_count} tokens streamed to file | {tokens_per_sec:.1f} tok/s | Elapsed: {int(elapsed)}s")
                                last_update = current_time
                                f.flush()  # Flush after status update

                    # Final flush before closing
                    f.flush()

                # Final progress update
                elapsed = time.time() - start_time
                # Read back for word count (file is already closed)
                with open(output_file, 'r', encoding='utf-8') as f:
                    overall_summary = f.read()
                word_count = len(overall_summary.split())
                print(f"✓ Final summary complete: {word_count} words, {token_count} tokens streamed to file in {int(elapsed)}s")
            except (OSError, IOError) as e:
                print(f"\n❌ File I/O error during summary streaming: {e}")
                print("Attempting to recover summary from file...")
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        overall_summary = f.read()
                    if overall_summary:
                        print(f"✓ Recovered {len(overall_summary.split())} words from partial summary")
                    else:
                        raise RuntimeError("Could not recover summary - file is empty")
                except Exception as recovery_error:
                    raise RuntimeError(f"Summary generation failed and recovery failed: {recovery_error}")

        else:
            # Traditional in-memory processing
            print("Generating final summary...")
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    overall_summary += content
                    token_count += 1

                    # Update progress every 5 seconds (less spam in GUI)
                    current_time = time.time()
                    if current_time - last_update >= 5.0:
                        elapsed = current_time - start_time
                        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                        print(f"Progress: {token_count} tokens generated | {tokens_per_sec:.1f} tok/s | Elapsed: {int(elapsed)}s")
                        last_update = current_time

            # Final progress update
            elapsed = time.time() - start_time
            word_count = len(overall_summary.split())
            print(f"✓ Final summary complete: {word_count} words, {token_count} tokens generated in {int(elapsed)}s")

        return overall_summary

    def process_multiple_audio_files(self, file_to_speaker: Dict[str, str], output_dir: str = None, session_name: str = None) -> Dict:
        """
        Process multiple audio files (one per speaker) and merge into single transcript

        Args:
            file_to_speaker: Dictionary mapping file paths to speaker names
            output_dir: Directory to save outputs (defaults to DEFAULT_OUTPUT_DIR)
            session_name: Name for this session (defaults to "multi_file_session")

        Returns:
            Dictionary with all outputs
        """
        import time

        # Use default output directory if none provided
        if output_dir is None:
            output_dir = str(DEFAULT_OUTPUT_DIR)

        # Use default session name if none provided
        if session_name is None:
            session_name = "multi_file_session"

        print("\n" + BANNER_SINGLE)
        print("MULTI-FILE PROCESSING MODE")
        print("="*60)
        print(f"Session: {session_name}")
        print(f"Processing {len(file_to_speaker)} audio files...")
        print("="*60 + "\n")

        total_start_time = time.time()
        timings = {}

        # Step 1: Transcribe each file
        print(f"\n{'='*60}")
        print(f"TRANSCRIPTION")
        print(f"{'='*60}")
        print(f"Transcribing {len(file_to_speaker)} files...")
        print(f"{'='*60}\n")

        transcription_start = time.time()
        file_transcripts = {}

        for i, (audio_file, speaker) in enumerate(file_to_speaker.items(), 1):
            print(f"\n[File {i}/{len(file_to_speaker)}] Transcribing {speaker}...")
            # Enable VAD in multi-file mode to skip silence in individual speaker tracks
            transcription_data = self.transcribe_audio(audio_file, use_vad=True)

            # Add speaker to each segment
            for seg in transcription_data["segments"]:
                seg["speaker"] = speaker

            file_transcripts[audio_file] = transcription_data

        timings['transcription'] = time.time() - transcription_start

        # Save individual speaker transcripts (raw + filtered)
        print(f"\n{'='*60}")
        print(f"SAVING INDIVIDUAL TRANSCRIPTS")
        print(f"{'='*60}\n")

        # Store session name for character tracking
        self.current_session_name = session_name
        # Ensure base output_dir exists first
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / session_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Create transcripts subdirectory
        transcripts_dir = output_path / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        for audio_file, transcript_data in file_transcripts.items():
            speaker = file_to_speaker[audio_file]

            # Save RAW transcript (before hallucination filter)
            raw_file = transcripts_dir / f"{speaker}_raw.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"RAW Whisper Transcript for {speaker}\n")
                f.write(f"(Before hallucination filtering)\n")
                f.write(f"File: {audio_file}\n")
                f.write(f"Segments: {len(transcript_data.get('raw_segments', []))}\n")
                f.write("="*80 + "\n\n")
                for seg in transcript_data.get('raw_segments', []):
                    start_ts = format_timestamp(seg['start'])
                    end_ts = format_timestamp(seg['end'])
                    f.write(f"[{start_ts} - {end_ts}] {seg['text'].strip()}\n")
            print(f"✓ Saved RAW transcript for {speaker}: {raw_file.name}")

            # Save FILTERED transcript (after hallucination filter)
            filtered_file = transcripts_dir / f"{speaker}_filtered.txt"
            with open(filtered_file, 'w', encoding='utf-8') as f:
                f.write(f"FILTERED Transcript for {speaker}\n")
                f.write(f"(After hallucination filtering)\n")
                f.write(f"File: {audio_file}\n")
                f.write(f"Segments: {len(transcript_data['segments'])}\n")
                removed = len(transcript_data.get('raw_segments', [])) - len(transcript_data['segments'])
                f.write(f"Removed segments: {removed}\n")
                f.write("="*80 + "\n\n")
                for seg in transcript_data['segments']:
                    start_ts = format_timestamp(seg['start'])
                    end_ts = format_timestamp(seg['end'])
                    f.write(f"[{start_ts} - {end_ts}] {seg['text'].strip()}\n")
            print(f"✓ Saved FILTERED transcript for {speaker}: {filtered_file.name}")

        # Step 2: Merge all transcripts by timestamp
        print(f"\n{'='*60}")
        print(f"MERGING TRANSCRIPTS")
        print(f"{'='*60}\n")

        all_segments = []
        for audio_file, transcript_data in file_transcripts.items():
            speaker = file_to_speaker[audio_file]
            # Add speaker info to each segment
            for seg in transcript_data["segments"]:
                seg["speaker"] = speaker
            all_segments.extend(transcript_data["segments"])

        # Sort by start time
        all_segments.sort(key=lambda x: x["start"])

        print(f"Initial segments: {len(all_segments)} from {len(file_to_speaker)} speakers")

        # Split overlapping segments to preserve conversation flow
        all_segments = self.split_overlapping_segments(all_segments)

        print(f"After splitting overlaps: {len(all_segments)} segments")

        # Combine into full text
        full_text = " ".join(seg["text"].strip() for seg in all_segments)

        # Output directory already created above when saving individual transcripts
        print(f"Output directory: {output_path}\n")

        # Save raw transcription
        transcript_file = output_path / f"{session_name}_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Raw transcript saved to: {transcript_file}")

        # Save formatted transcription with timestamps and speakers
        formatted_transcript_file = output_path / f"{session_name}_transcript_formatted.txt"
        with open(formatted_transcript_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"TTRPG SESSION TRANSCRIPT (Multi-File Mode)\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {all_segments[-1]['end']:.1f} seconds ({all_segments[-1]['end']/60:.1f} minutes)\n")
            f.write(f"Speakers: {', '.join(set(file_to_speaker.values()))}\n")
            f.write(f"Files processed: {len(file_to_speaker)}\n")
            f.write("="*80 + "\n\n")

            # Merge consecutive segments from same speaker
            merged_segments = self.merge_consecutive_speaker_segments(all_segments)

            for segment in merged_segments:
                # Format timestamps using module-level function
                start_ts = format_timestamp(segment['start'])
                end_ts = format_timestamp(segment['end'])
                timestamp = f"[{start_ts} - {end_ts}]"
                speaker_label = f"{segment['speaker']}: "
                f.write(f"{timestamp} {speaker_label}\n{segment['text'].strip()}\n\n")

        print(f"Formatted transcript saved to: {formatted_transcript_file}")

        # Free up GPU memory before summarization
        print("\nFreeing GPU memory for Ollama...")
        del self.whisper_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
        print("GPU memory freed! Ready for summarization.\n")

        # Get campaign context
        campaign_context = self.get_campaign_context()

        # Chunk the text
        chunks = self.chunk_text(full_text)

        # Summarize each chunk
        print(f"\n{'='*60}")
        print(f"CHUNK SUMMARIZATION (with progressive context)")
        print(f"{'='*60}")
        print(f"Total chunks to process: {len(chunks)}")
        print(f"{'='*60}\n")

        summarization_start = time.time()
        chunk_summaries = []
        progressive_context = ""  # Build context as we go

        for i, chunk in enumerate(chunks, 1):
            # Check for stop/pause before processing each chunk
            self.check_control_events()

            # Summarize chunk with context from previous chunks
            summary = self.summarize_chunk(chunk, i, len(chunks), context=campaign_context, previous_context=progressive_context)
            chunk_summaries.append(summary)
            # Create short context from this summary and add to progressive context
            if i < len(chunks):  # Don't build context after last chunk
                print(f"  Building context for next chunk...", end='', flush=True)
                short_context = self.create_short_context(summary)
            else:
                short_context = None
            if short_context:
                if progressive_context:
                    progressive_context += f"\n{short_context}"
                else:
                    progressive_context = short_context
                print(" ✓")

        # Save chunk summaries
        chunk_summaries_file = output_path / f"{session_name}_chunk_summaries.json"
        with open(chunk_summaries_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_summaries, f, indent=2)
        print(f"Chunk summaries saved to: {chunk_summaries_file}")

        # Save human-readable chunk summaries
        chunk_summaries_readable = output_path / f"{session_name}_chunk_summaries_readable.txt"
        with open(chunk_summaries_readable, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CHUNK SUMMARIES - HUMAN READABLE\n")
            f.write("="*80 + "\n\n")

            for i, summary in enumerate(chunk_summaries, 1):
                # Clean up the text - remove markdown formatting
                clean_summary = summary.replace('**', '').replace('###', '').replace('##', '#')
                # Ensure proper paragraph spacing (replace single newlines with double)
                clean_summary = clean_summary.replace('\n\n', '<<PARA>>').replace('\n', ' ').replace('<<PARA>>', '\n\n')

                f.write(f"{'='*80}\n")
                f.write(f"CHUNK {i} of {len(chunk_summaries)}\n")
                f.write(f"{'='*80}\n\n")
                f.write(clean_summary.strip())
                f.write("\n\n")

        print(f"Human-readable chunk summaries saved to: {chunk_summaries_readable}")

        # Check if final summary settings are pre-configured
        if self.preconfigured_final_summary:
            print("\n" + "="*60)
            print("Using pre-configured final summary settings...")
            print("="*60)

            model = self.ollama_final_model
            target_words = self.final_summary_target_words
            context_size = self.final_context_size if self.final_context_size else 32768

            print(f"\n✓ Pre-configured settings:")
            print(f"  Model: {model}")
            print(f"  Target words: {target_words}")
            print(f"  Context size: {'Unlimited' if context_size >= 999999 else f'{context_size:,} tokens'}")
        else:
            # Show final summary configuration dialog
            print("\n" + "="*60)
            print("Opening final summary configuration dialog...")
            print("="*60)

            config = self.show_final_summary_config(
                chunk_summaries,
                default_target_words=self.final_summary_target_words,
                default_context_size=None  # Let popup calculate smart default
            )

            if config is None:
                print("\n⚠️ Final summary generation cancelled by user")
                print("Chunk summaries have been saved. You can generate the final summary later.")
                return None

            model, target_words, context_size, include_chars = config
            print(f"\n✓ User confirmed settings:")
            print(f"  Model: {model}")
            print(f"  Target words: {target_words}")
            print(f"  Context size: {'Unlimited' if context_size >= 999999 else f'{context_size:,} tokens'}")
            print(f"  Include characters: {'Yes' if include_chars else 'No'}")

            # Update instance settings for this run
            self.ollama_final_model = model
            self.final_summary_target_words = target_words

        # Prepare summary filename
        model_safe = model.replace(':', '_').replace('/', '_')  # Make model name filesystem-safe
        summary_file = output_path / f"{session_name}_summary_{model_safe}.txt"

        # Create overall summary (stream directly to file to reduce memory usage)
        overall_summary = self.create_overall_summary(
            chunk_summaries,
            context=campaign_context,
            final_context_size=context_size,
            output_file=str(summary_file),
            include_characters=include_chars
        )
        timings['summarization'] = time.time() - summarization_start
        print(f"✓ Overall summary saved to: {summary_file}")

        # Save character/NPC list
        print(f"\n[Debug] Checking characters before save: {len(self.characters)} characters tracked")
        print(f"[Debug] Campaign file path: {self.campaign_character_file}")
        if self.characters:
            if self.campaign_character_file:
                # Update campaign file
                self.save_campaign_characters()
            else:
                # Save session-specific file (legacy behavior)
                characters_file = output_path / f"{session_name}_characters.txt"
                with open(characters_file, 'w', encoding='utf-8') as f:
                    f.write(f"{'='*60}\n")
                    f.write(f"CHARACTERS & NPCs - {session_name}\n")
                    f.write(f"{'='*60}\n\n")

                    # Sort by first appearance
                    sorted_chars = sorted(self.characters.items(),
                                        key=lambda x: x[1]['first_appearance'])

                    for name, info in sorted_chars:
                        f.write(f"**{name}**\n")
                        f.write(f"  First appeared: {info['first_appearance']}\n")
                        f.write(f"  Mentioned: {info['mentions']} time(s)\n")
                        f.write(f"  Description: {info['description']}\n\n")

                print(f"✓ Character list saved to: {characters_file}")
                print(f"  Total characters/NPCs tracked: {len(self.characters)}")

        # Clean up temporary WAV file if it wasn't already cleaned up (happens when diarization is disabled)
        temp_wav = transcription_data.get("temp_wav_file")
        if temp_wav and Path(temp_wav).exists():
            try:
                import os
                os.unlink(temp_wav)
                print(f"✓ Cleaned up temporary WAV file")
            except Exception as e:
                print(f"Warning: Could not delete temp WAV file: {e}")

        # Calculate total time
        total_time = time.time() - total_start_time
        timings['total'] = total_time

        # Print timing summary (using module-level format_time function)
        print(f"\n{BANNER_SINGLE}")
        print(f"PROCESSING TIME SUMMARY")
        print(f"{'='*60}")
        print(f"Transcription:        {format_time(timings['transcription'])}")
        print(f"Summarization:        {format_time(timings['summarization'])}")
        print(f"{'-'*60}")
        print(f"TOTAL TIME:           {format_time(timings['total'])}")
        print(f"{'='*60}\n")

        # Unload Ollama model from VRAM
        print("\nUnloading Ollama model from VRAM...")
        try:
            import subprocess
            subprocess.run(["ollama", "stop", self.ollama_model], capture_output=True)
            print("Ollama model unloaded successfully!")
        except Exception as e:
            print(f"Note: Could not unload Ollama model (it will auto-unload after 5 min): {e}")

        result = {
            "transcript": full_text,
            "transcript_file": str(transcript_file),
            "formatted_transcript_file": str(formatted_transcript_file),
            "chunk_summaries": chunk_summaries,
            "chunk_summaries_file": str(chunk_summaries_file),
            "overall_summary": overall_summary,
            "summary_file": str(summary_file),
            "timestamp": datetime.now().isoformat(),
            "timings": timings,
            "mode": "multi-file",
            "num_files": len(file_to_speaker),
            "speakers": list(set(file_to_speaker.values()))
        }

        return result

    def process_audio_file(self, audio_file: str, output_dir: str = None) -> Dict:
        """
        Complete pipeline: audio -> text -> summaries

        Args:
            audio_file: Path to audio file
            output_dir: Directory to save outputs (defaults to DEFAULT_OUTPUT_DIR)

        Returns:
            Dictionary with all outputs
        """
        import time

        # Use default output directory if none provided
        if output_dir is None:
            output_dir = str(DEFAULT_OUTPUT_DIR)

        # Track timing for each stage
        timings = {}
        total_start_time = time.time()

        # Create output directory with subfolder named after the audio file
        audio_name = Path(audio_file).stem
        # Store session name for character tracking
        self.current_session_name = audio_name
        # Ensure base output_dir exists first
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / audio_name
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")

        # Step 1: Speaker diarization (if enabled)
        diarization_data = None
        speaker_map = None
        aligned_segments = None

        if self.enable_diarization:
            print(f"\n{BANNER_SINGLE}")
            print("STEP 1: SPEAKER DIARIZATION")
            print(f"{BANNER_SINGLE}")
            diarization_start = time.time()
            diarization_data = self.perform_diarization(audio_file)
            timings['diarization'] = time.time() - diarization_start
            print(f"✓ Diarization complete\n")

        # Step 2: Transcribe audio
        print(f"\n{BANNER_SINGLE}")
        print("STEP 2: AUDIO TRANSCRIPTION" if self.enable_diarization else "STEP 1: AUDIO TRANSCRIPTION")
        print(f"{BANNER_SINGLE}")
        transcription_start = time.time()
        transcription_data = self.transcribe_audio(audio_file)
        full_text = transcription_data["text"]
        segments = transcription_data["segments"]
        timings['transcription'] = time.time() - transcription_start

        # Step 3: Align diarization with transcription (if enabled)
        # Note: If diarization is disabled, temp WAV will be cleaned up later after chunks are saved
        if self.enable_diarization and diarization_data:
            print(f"\n{BANNER_SINGLE}")
            print("STEP 3: ALIGNING SPEAKERS WITH TRANSCRIPT")
            print(f"{BANNER_SINGLE}")
            aligned_segments = self.align_diarization_with_transcription(
                transcription_data, diarization_data
            )

            # Fix UNKNOWN speakers by merging with adjacent speakers
            print("Cleaning up UNKNOWN speaker labels...")
            unknown_count_before = sum(1 for seg in aligned_segments if seg.get("speaker") == "UNKNOWN")
            aligned_segments = self.fix_unknown_speakers(aligned_segments)
            unknown_count_after = sum(1 for seg in aligned_segments if seg.get("speaker") == "UNKNOWN")
            if unknown_count_before > unknown_count_after:
                print(f"  Fixed {unknown_count_before - unknown_count_after} UNKNOWN segments")

            # Ask user to map speaker names (NOW with audio playback!)
            # Use WAV file for fast clip extraction (if available)
            audio_for_clips = diarization_data.get("wav_file", audio_file)
            speaker_map = self.map_speaker_names_with_samples(
                diarization_data["speakers"],
                aligned_segments,
                audio_file=audio_for_clips  # Use WAV file for fast seeking
            )

            # Apply speaker name mapping
            for seg in aligned_segments:
                seg["speaker"] = speaker_map.get(seg["speaker"], seg["speaker"])

            # Save speaker mapping
            speaker_map_file = output_path / f"{audio_name}_speakers.json"
            with open(speaker_map_file, 'w', encoding='utf-8') as f:
                json.dump(speaker_map, f, indent=2)
            print(f"Speaker mapping saved to: {speaker_map_file}")

            # Clean up temporary WAV files now that speaker clips have been extracted
            temp_wav = transcription_data.get("temp_wav_file")
            if temp_wav and Path(temp_wav).exists():
                try:
                    import os
                    os.unlink(temp_wav)
                    print(f"✓ Cleaned up temporary transcription WAV file")
                except Exception as e:
                    print(f"Warning: Could not delete temp transcription WAV file: {e}")

            # Also clean up diarization temp WAV if it exists
            if diarization_data and diarization_data.get("is_temp_wav"):
                diar_wav = diarization_data.get("wav_file")
                if diar_wav and Path(diar_wav).exists():
                    try:
                        import os
                        os.unlink(diar_wav)
                        print(f"✓ Cleaned up temporary diarization WAV file")
                    except Exception as e:
                        print(f"Warning: Could not delete temp diarization WAV file: {e}")

        # Save raw transcription
        transcript_file = output_path / f"{audio_name}_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Raw transcript saved to: {transcript_file}")

        # Save formatted transcription with timestamps (and speakers if available)
        formatted_transcript_file = output_path / f"{audio_name}_transcript_formatted.txt"
        with open(formatted_transcript_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"TTRPG SESSION TRANSCRIPT\n")
            f.write(f"Audio File: {Path(audio_file).name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {segments[-1]['end']:.1f} seconds ({segments[-1]['end']/60:.1f} minutes)\n")
            if self.enable_diarization and speaker_map:
                f.write(f"Speakers: {', '.join(speaker_map.values())}\n")
            f.write("="*80 + "\n\n")

            # Use aligned segments if available, otherwise use regular segments
            segments_to_write = aligned_segments if aligned_segments else segments

            # Merge consecutive segments from the same speaker for readability
            if aligned_segments:
                segments_to_write = self.merge_consecutive_speaker_segments(segments_to_write)

            for segment in segments_to_write:
                # Format timestamps using module-level function
                start_ts = format_timestamp(segment['start'])
                end_ts = format_timestamp(segment['end'])
                timestamp = f"[{start_ts} - {end_ts}]"
                speaker_label = f"{segment['speaker']}: " if 'speaker' in segment else ""
                f.write(f"{timestamp} {speaker_label}\n{segment['text'].strip()}\n\n")

        print(f"Formatted transcript saved to: {formatted_transcript_file}")

        # Free up GPU memory before summarization
        print("\nFreeing GPU memory for Ollama...")
        del self.whisper_model
        if self.enable_diarization and self.diarization_pipeline:
            del self.diarization_pipeline
            self.diarization_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
        print("GPU memory freed! Ready for summarization.\n")

        # Get campaign context from user before summarization
        campaign_context = self.get_campaign_context()

        # Step 2: Chunk the text
        step_num = 4 if self.enable_diarization else 2
        print(f"\n{BANNER_SINGLE}")
        print(f"STEP {step_num}: CHUNKING TEXT FOR SUMMARIZATION")
        print(f"{BANNER_SINGLE}")
        chunks = self.chunk_text(full_text)
        print(f"✓ Created {len(chunks)} chunks\n")

        # Step 3: Summarize each chunk
        step_num = 5 if self.enable_diarization else 3
        print(f"\n{'='*60}")
        print(f"STEP {step_num}: CHUNK SUMMARIZATION (with progressive context)")
        print(f"{'='*60}")
        print(f"Total chunks to process: {len(chunks)}")
        print(f"{'='*60}\n")

        summarization_start = time.time()
        chunk_summaries = []
        progressive_context = ""  # Build context as we go

        for i, chunk in enumerate(chunks, 1):
            # Check for stop/pause before processing each chunk
            self.check_control_events()

            # Summarize chunk with context from previous chunks
            summary = self.summarize_chunk(chunk, i, len(chunks), context=campaign_context, previous_context=progressive_context)
            chunk_summaries.append(summary)
            # Create short context from this summary and add to progressive context
            if i < len(chunks):  # Don't build context after last chunk
                print(f"  Building context for next chunk...", end='', flush=True)
                short_context = self.create_short_context(summary)
            else:
                short_context = None
            if short_context:
                if progressive_context:
                    progressive_context += f"\n{short_context}"
                else:
                    progressive_context = short_context
                print(" ✓")

        # Save chunk summaries
        chunk_summaries_file = output_path / f"{audio_name}_chunk_summaries.json"
        with open(chunk_summaries_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_summaries, f, indent=2)
        print(f"Chunk summaries saved to: {chunk_summaries_file}")

        # Save human-readable chunk summaries
        chunk_summaries_readable = output_path / f"{audio_name}_chunk_summaries_readable.txt"
        with open(chunk_summaries_readable, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CHUNK SUMMARIES - HUMAN READABLE\n")
            f.write("="*80 + "\n\n")

            for i, summary in enumerate(chunk_summaries, 1):
                # Clean up the text - remove markdown formatting
                clean_summary = summary.replace('**', '').replace('###', '').replace('##', '#')
                # Ensure proper paragraph spacing (replace single newlines with double)
                clean_summary = clean_summary.replace('\n\n', '<<PARA>>').replace('\n', ' ').replace('<<PARA>>', '\n\n')

                f.write(f"{'='*80}\n")
                f.write(f"CHUNK {i} of {len(chunk_summaries)}\n")
                f.write(f"{'='*80}\n\n")
                f.write(clean_summary.strip())
                f.write("\n\n")

        print(f"Human-readable chunk summaries saved to: {chunk_summaries_readable}")

        # Check if final summary settings are pre-configured
        if self.preconfigured_final_summary:
            print("\n" + "="*60)
            print("Using pre-configured final summary settings...")
            print("="*60)

            model = self.ollama_final_model
            target_words = self.final_summary_target_words
            context_size = self.final_context_size if self.final_context_size else 32768

            print(f"\n✓ Pre-configured settings:")
            print(f"  Model: {model}")
            print(f"  Target words: {target_words}")
            print(f"  Context size: {'Unlimited' if context_size >= 999999 else f'{context_size:,} tokens'}")
        else:
            # Show final summary configuration dialog
            print("\n" + "="*60)
            print("Opening final summary configuration dialog...")
            print("="*60)

            config = self.show_final_summary_config(
                chunk_summaries,
                default_target_words=self.final_summary_target_words,
                default_context_size=None  # Let popup calculate smart default
            )

            if config is None:
                print("\n⚠️ Final summary generation cancelled by user")
                print("Chunk summaries have been saved. You can generate the final summary later.")
                return None

            model, target_words, context_size, include_chars = config
            print(f"\n✓ User confirmed settings:")
            print(f"  Model: {model}")
            print(f"  Target words: {target_words}")
            print(f"  Context size: {'Unlimited' if context_size >= 999999 else f'{context_size:,} tokens'}")
            print(f"  Include characters: {'Yes' if include_chars else 'No'}")

            # Update instance settings for this run
            self.ollama_final_model = model
            self.final_summary_target_words = target_words

        # Prepare summary filename
        model_safe = model.replace(':', '_').replace('/', '_')  # Make model name filesystem-safe
        summary_file = output_path / f"{audio_name}_summary_{model_safe}.txt"

        # Step 4: Create overall summary (stream directly to file to reduce memory usage)
        step_num = 6 if self.enable_diarization else 4
        print(f"\n{BANNER_SINGLE}")
        print(f"STEP {step_num}: GENERATING FINAL SUMMARY")
        print(f"{BANNER_SINGLE}")
        print(f"Combining {len(chunk_summaries)} chunk summaries into final narrative...\n")
        overall_summary = self.create_overall_summary(
            chunk_summaries,
            context=campaign_context,
            final_context_size=context_size,
            output_file=str(summary_file),
            include_characters=include_chars
        )
        timings['summarization'] = time.time() - summarization_start
        print(f"✓ Overall summary saved to: {summary_file}")

        # Save character/NPC list
        print(f"\n[Debug] Checking characters before save: {len(self.characters)} characters tracked")
        print(f"[Debug] Campaign file path: {self.campaign_character_file}")
        if self.characters:
            if self.campaign_character_file:
                # Update campaign file
                self.save_campaign_characters()
            else:
                # Save session-specific file (legacy behavior)
                characters_file = output_path / f"{audio_name}_characters.txt"
                with open(characters_file, 'w', encoding='utf-8') as f:
                    f.write(f"{'='*60}\n")
                    f.write(f"CHARACTERS & NPCs - {audio_name}\n")
                    f.write(f"{'='*60}\n\n")

                    # Sort by first appearance
                    sorted_chars = sorted(self.characters.items(),
                                        key=lambda x: x[1]['first_appearance'])

                    for name, info in sorted_chars:
                        f.write(f"**{name}**\n")
                        f.write(f"  First appeared: {info['first_appearance']}\n")
                        f.write(f"  Mentioned: {info['mentions']} time(s)\n")
                        f.write(f"  Description: {info['description']}\n\n")

                print(f"✓ Character list saved to: {characters_file}")
                print(f"  Total characters/NPCs tracked: {len(self.characters)}")

        # Clean up temporary WAV file if it wasn't already cleaned up (happens when diarization is disabled)
        temp_wav = transcription_data.get("temp_wav_file")
        if temp_wav and Path(temp_wav).exists():
            try:
                import os
                os.unlink(temp_wav)
                print(f"✓ Cleaned up temporary WAV file")
            except Exception as e:
                print(f"Warning: Could not delete temp WAV file: {e}")

        # Return all data
        result = {
            "audio_file": audio_file,
            "transcript": full_text,
            "transcript_file": str(transcript_file),
            "formatted_transcript_file": str(formatted_transcript_file),
            "chunk_summaries": chunk_summaries,
            "chunk_summaries_file": str(chunk_summaries_file),
            "overall_summary": overall_summary,
            "summary_file": str(summary_file),
            "timestamp": transcription_data["timestamp"]
        }

        # Add speaker data if diarization was enabled
        if self.enable_diarization and speaker_map:
            result["speaker_map_file"] = str(output_path / f"{audio_name}_speakers.json")
            result["speaker_map"] = speaker_map

        # Calculate total time
        total_time = time.time() - total_start_time
        timings['total'] = total_time

        # Print timing summary (using module-level format_time function)
        print(f"\n{BANNER_SINGLE}")
        print(f"PROCESSING TIME SUMMARY")
        print(f"{'='*60}")
        if 'diarization' in timings:
            print(f"Speaker Diarization:  {format_time(timings['diarization'])}")
        print(f"Transcription:        {format_time(timings['transcription'])}")
        print(f"Summarization:        {format_time(timings['summarization'])}")
        print(f"{'-'*60}")
        print(f"TOTAL TIME:           {format_time(timings['total'])}")
        print(f"{'='*60}\n")

        # Add timings to result
        result["timings"] = timings

        # Unload Ollama model from VRAM
        print("\nUnloading Ollama model from VRAM...")
        try:
            import subprocess
            subprocess.run(["ollama", "stop", self.ollama_model], capture_output=True)
            print("Ollama model unloaded successfully!")
        except Exception as e:
            print(f"Note: Could not unload Ollama model (it will auto-unload after 5 min): {e}")

        return result


def select_processing_mode() -> str:
    """
    Ask user whether to process single file or multiple files

    Returns:
        'single' or 'multi'
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    mode = messagebox.askquestion(
        "Processing Mode",
        "Do you have MULTIPLE audio files (one per speaker)?\n\n"
        "Examples:\n"
        "- YES: Discord Craig bot recordings (separate file per person)\n"
        "- NO: Single recording with multiple speakers mixed together\n\n"
        "Click 'Yes' for multiple files, 'No' for single file"
    )

    root.destroy()
    return 'multi' if mode == 'yes' else 'single'


def select_audio_file() -> str:
    """
    Open a GUI file picker to select an audio file

    Returns:
        Path to selected audio file, or None if cancelled
    """
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    # Open file dialog
    audio_file = filedialog.askopenfilename(
        title="Select Audio File to Transcribe",
        filetypes=[
            ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac"),
            ("MP3 Files", "*.mp3"),
            ("WAV Files", "*.wav"),
            ("M4A Files", "*.m4a"),
            ("All Files", "*.*")
        ],
        initialdir=Path.home()
    )

    root.destroy()  # Clean up
    return audio_file


def select_multiple_audio_files() -> List[str]:
    """
    Open a GUI file picker to select multiple audio files

    Returns:
        List of paths to selected audio files, or empty list if cancelled
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    audio_files = filedialog.askopenfilenames(
        title="Select Audio Files (one per speaker)",
        filetypes=[
            ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("All Files", "*.*")
        ],
        initialdir=Path.home()
    )

    root.destroy()
    return list(audio_files) if audio_files else []


def assign_speakers_to_files(audio_files: List[str]) -> Dict[str, str]:
    """
    Ask user to assign speaker names to each audio file

    Args:
        audio_files: List of audio file paths

    Returns:
        Dictionary mapping file path to speaker name
    """
    print("\n" + "="*60)
    print("SPEAKER ASSIGNMENT")
    print("="*60)
    print(f"Assigning speakers to {len(audio_files)} audio files...")
    print("="*60 + "\n")

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_to_speaker = {}

    for audio_file in audio_files:
        filename = Path(audio_file).name

        # Try to auto-detect speaker from Craig bot naming pattern
        # Craig formats:
        #   - "number-username.aac" (e.g., "1-nic_231.aac")
        #   - "username_channelid.wav" (e.g., "nic_231_12345.wav")
        #   - "username.wav"
        suggested_name = ""
        stem = Path(audio_file).stem  # filename without extension

        # Check for Craig's "number-username" format (e.g., "1-nic_231")
        if '-' in stem and stem.split('-')[0].isdigit():
            # Extract username after the dash
            parts = stem.split('-', 1)  # Split on first dash only
            if len(parts) > 1:
                suggested_name = parts[1]  # "nic_231"
        elif '_' in stem:
            # Extract username before first underscore
            suggested_name = stem.split('_')[0]
        else:
            # Use filename without extension
            suggested_name = stem

        speaker_name = simpledialog.askstring(
            "Assign Speaker",
            f"File: {filename}\n\n"
            f"Who is this speaker?\n\n"
            f"Suggested: {suggested_name}\n"
            f"(Press OK to use suggestion, or type a different name)",
            initialvalue=suggested_name,
            parent=root
        )

        if speaker_name and speaker_name.strip():
            file_to_speaker[audio_file] = speaker_name.strip()
        else:
            # Use suggested name if user just pressed OK
            file_to_speaker[audio_file] = suggested_name

    root.destroy()

    print("\nSpeaker assignments:")
    for file_path, speaker in file_to_speaker.items():
        print(f"  {Path(file_path).name} → {speaker}")
    print()

    return file_to_speaker


def get_ollama_models():
    """Get list of installed Ollama models with size information"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0]
                        size = parts[2] if len(parts) > 2 else "Unknown"
                        models.append((name, size))
            return models
        return []
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to get Ollama models: {e}")
        return []


def get_ollama_model_info(model_name):
    """
    Query Ollama API for actual model information

    Returns:
        dict with 'size_bytes', 'parameter_size', 'quantization' or None if unavailable
    """
    try:
        # Use ollama.show() to get model details
        info = ollama.show(model_name)

        model_info = {
            'size_bytes': 0,
            'parameter_size': 0,
            'quantization': 'Q4_0'  # Default assumption
        }

        # Extract actual model file size if available
        if 'details' in info and 'parameter_size' in info['details']:
            param_str = info['details']['parameter_size']
            # Parse strings like "7B", "14B", "1.8B"
            import re
            match = re.search(r'(\d+\.?\d*)B', param_str, re.IGNORECASE)
            if match:
                model_info['parameter_size'] = float(match.group(1))

        # Extract quantization level if available
        if 'details' in info and 'quantization_level' in info['details']:
            model_info['quantization'] = info['details']['quantization_level']

        # Try to get actual size from modelfile or other fields
        if 'size' in info:
            model_info['size_bytes'] = info['size']

        return model_info

    except Exception:
        # If API call fails, return None to fall back to heuristic estimation
        return None


def estimate_vram_usage(model_name, context_size=8192):
    """
    Estimate VRAM usage for an Ollama model using actual model info when possible
    """
    import re

    # Try to get actual model info from Ollama API first
    model_info = get_ollama_model_info(model_name)

    if model_info and model_info['parameter_size'] > 0:
        # Use actual parameter size from API
        size_b = model_info['parameter_size']
        quant = model_info['quantization']

        # Determine bits per parameter based on quantization
        if 'Q2' in quant or 'q2' in model_name.lower():
            bits_per_param = 2.5  # Q2 is ~2.5 bits/param with overhead
        elif 'Q4' in quant or 'q4' in model_name.lower():
            bits_per_param = 4.5  # Q4 is ~4.5 bits/param with overhead
        elif 'Q5' in quant or 'q5' in model_name.lower():
            bits_per_param = 5.5
        elif 'Q6' in quant or 'q6' in model_name.lower():
            bits_per_param = 6.5
        elif 'Q8' in quant or 'q8' in model_name.lower():
            bits_per_param = 8.5
        else:
            bits_per_param = 4.5  # Default to Q4

        # Calculate base model VRAM: (parameters * bits_per_param) / 8 / 1024^3
        base_vram = (size_b * 1e9 * bits_per_param) / 8 / (1024**3)

    else:
        # Fall back to heuristic estimation from model name
        match = re.search(r'(\d+\.?\d*)b', model_name.lower())

        if not match:
            # Infer from common model names
            if 'llama3.2' in model_name.lower() or 'llama3' in model_name.lower():
                size_b = 8.0
            elif 'llama2' in model_name.lower():
                size_b = 7.0
            elif 'qwen' in model_name.lower():
                size_b = 14.0
            else:
                size_b = 7.0  # Conservative default
        else:
            size_b = float(match.group(1))

        # Estimate base VRAM using heuristics
        if 'q2' in model_name.lower():
            base_vram = size_b * 0.35
        elif 'q4' in model_name.lower() or ':' in model_name:
            base_vram = size_b * 0.6
        elif 'q8' in model_name.lower():
            base_vram = size_b * 1.1
        else:
            base_vram = size_b * 0.6

    # Add context overhead (scales with model size and context length)
    # KV cache for transformers: 2 (key+value) * num_layers * hidden_dim * context_length * bytes_per_element
    # Based on real-world testing with qwen2.5:14b @ 131K context = ~16GB total:
    # - Base model: ~8.4 GB (no overhead yet)
    # - Context overhead: ~7.6 GB (no overhead yet)
    # - That's (131.072) * 14 * X = 7.6 GB → X = 0.0414
    # Calibrated formula: ~0.041 GB per 1K tokens per billion parameters
    context_vram = (context_size / 1000) * size_b * 0.041

    # Total already includes overhead, no additional multiplier needed
    total = base_vram + context_vram

    return f"~{total:.1f} GB"


def get_safe_context_size(model_name, requested_context_size):
    """
    Check if requested context size would exceed available VRAM.
    If it would, return None (unlimited) to let Ollama manage memory dynamically.
    Otherwise, return the requested context size.

    Args:
        model_name: Name of the Ollama model
        requested_context_size: Desired context size in tokens (or 999999 for unlimited)

    Returns:
        Context size to use, or None for unlimited
    """
    # If already unlimited, pass through as None
    if requested_context_size >= 999999:
        return None

    # Get available VRAM
    _, available_vram_gb, _ = get_available_vram()

    # If no GPU or can't detect, use the requested size (best effort)
    if available_vram_gb <= 0:
        return requested_context_size

    # Estimate VRAM usage for this context size
    vram_estimate_str = estimate_vram_usage(model_name, requested_context_size)
    try:
        # Parse the estimate (format: "~12.5 GB")
        estimated_vram_gb = float(vram_estimate_str.replace("~", "").replace("GB", "").strip())

        # If estimated usage exceeds available VRAM, convert to unlimited
        if estimated_vram_gb > available_vram_gb:
            print(f"⚠️  Context size {requested_context_size:,} tokens would use ~{estimated_vram_gb:.1f} GB VRAM")
            print(f"   Available VRAM: {available_vram_gb:.1f} GB")
            print(f"   Converting to UNLIMITED context to let Ollama manage memory dynamically")
            print(f"   (This prevents crashes and allows graceful overflow to shared GPU memory)")
            return None  # Unlimited
        else:
            return requested_context_size

    except (ValueError, AttributeError):
        # If we can't parse the estimate, use the requested size
        return requested_context_size


def configuration_gui():
    """
    Main configuration GUI for TTRPG Summarizer

    Returns:
        Dictionary with all configuration options, or None if cancelled
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    config = [None]  # Use list to store result

    # Create main window
    root = tk.Tk()
    root.title("🎲 TTRPG Summarizer - Configuration")
    root.geometry("1400x1000")  # Increased height from 900 to 1000 to show all controls
    root.configure(bg="#ecf0f1")

    # Try to set a custom icon (Windows only - falls back gracefully on other platforms)
    try:
        # Create a simple icon using PIL if available
        from PIL import Image, ImageDraw, ImageTk

        # Create a 32x32 icon with a D20 dice design
        icon_size = 32
        img = Image.new('RGBA', (icon_size, icon_size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # Draw a purple hexagon (simplified D20 representation)
        points = [
            (16, 2), (28, 10), (28, 22), (16, 30), (4, 22), (4, 10)
        ]
        draw.polygon(points, fill='#9b59b6', outline='#2c3e50')

        # Draw "20" in the center
        draw.text((8, 10), "20", fill='white')

        # Convert to PhotoImage and set as icon
        photo = ImageTk.PhotoImage(img)
        root.iconphoto(True, photo)
    except Exception:
        # If icon creation fails, just use the emoji in title (already done above)
        pass

    # Main frame with padding
    main_frame = tk.Frame(root, bg="#ecf0f1", padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Header
    header = tk.Frame(main_frame, bg="#2c3e50", padx=15, pady=15)
    header.pack(fill=tk.X, pady=(0, 20))
    tk.Label(header, text="TTRPG Session Summarizer", font=("Arial", 20, "bold"),
             bg="#2c3e50", fg="white").pack()
    tk.Label(header, text="Configure your transcription and summarization settings",
             font=("Arial", 10), bg="#2c3e50", fg="#ecf0f1").pack()

    # Content area - split into left (config) and right (console)
    content_frame = tk.Frame(main_frame, bg="#ecf0f1")
    content_frame.pack(fill=tk.BOTH, expand=True)

    # LEFT SIDE: Scrollable frame for options
    left_frame = tk.Frame(content_frame, bg="#ecf0f1")
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

    canvas = tk.Canvas(left_frame, bg="#ecf0f1", highlightthickness=0, width=500)
    scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
    scrollable = tk.Frame(canvas, bg="#ecf0f1")

    scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Enable mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # Bind mouse wheel to canvas and all child widgets
    def bind_mousewheel(widget):
        widget.bind("<MouseWheel>", on_mousewheel)
        for child in widget.winfo_children():
            bind_mousewheel(child)

    bind_mousewheel(root)
    canvas.bind("<MouseWheel>", on_mousewheel)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # RIGHT SIDE: Console output area
    right_frame = tk.Frame(content_frame, bg="white", padx=10, pady=10)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

    # Header with Console Output label and Token button
    header_frame = tk.Frame(right_frame, bg="white")
    header_frame.pack(fill=tk.X, pady=(0, 5))

    console_label = tk.Label(header_frame, text="Console Output", font=("Arial", 12, "bold"),
                            bg="white", fg="#2c3e50")
    console_label.pack(side=tk.LEFT)

    def open_token_window():
        """Open token management window"""
        token_window = tk.Toplevel(root)
        token_window.title("HuggingFace Token Management")
        token_window.geometry("550x400")
        token_window.configure(bg="#ecf0f1")
        token_window.attributes('-topmost', True)

        # Header
        header = tk.Frame(token_window, bg="#3498db", padx=20, pady=15)
        header.pack(fill=tk.X)
        tk.Label(header, text="🔑 HuggingFace Token", font=("Arial", 16, "bold"),
                bg="#3498db", fg="white").pack()
        tk.Label(header, text="Required for Speaker Diarization",
                font=("Arial", 10), bg="#3498db", fg="#ecf0f1").pack()

        # Content
        content = tk.Frame(token_window, bg="white", padx=20, pady=20)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Info
        info_text = ("To use speaker diarization, you need a HuggingFace token with access to:\n"
                    "  • pyannote/speaker-diarization-3.1\n"
                    "  • pyannote/segmentation-3.0\n\n"
                    "Get your token at: https://huggingface.co/settings/tokens")
        tk.Label(content, text=info_text, bg="white", font=("Arial", 9),
                fg="#7f8c8d", justify=tk.LEFT, wraplength=480).pack(anchor=tk.W, pady=(0, 15))

        # Current status
        status_frame = tk.LabelFrame(content, text="Current Status", bg="white",
                                    font=("Arial", 10, "bold"), padx=15, pady=10)
        status_frame.pack(fill=tk.X, pady=(0, 15))

        status_text = "✓ Token saved and ready" if saved_token else "✗ No token saved"
        status_color = "#27ae60" if saved_token else "#e74c3c"
        tk.Label(status_frame, text=status_text, bg="white",
                font=("Arial", 10, "bold"), fg=status_color).pack(anchor=tk.W)

        if saved_token:
            tk.Label(status_frame, text=f"Token: ●●●●●●●●{saved_token[-4:]}",
                    bg="white", font=("Arial", 9), fg="#7f8c8d").pack(anchor=tk.W, pady=(5, 0))

        # Token entry
        entry_frame = tk.LabelFrame(content, text="Manage Token", bg="white",
                                   font=("Arial", 10, "bold"), padx=15, pady=10)
        entry_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(entry_frame, text="Token:", bg="white", font=("Arial", 9)).pack(anchor=tk.W, pady=(0, 5))
        token_display = tk.Entry(entry_frame, textvariable=hf_token_var, font=("Arial", 10),
                               width=40, show="●", state='readonly')
        token_display.pack(fill=tk.X, pady=(0, 10))

        button_frame = tk.Frame(entry_frame, bg="white")
        button_frame.pack(fill=tk.X)

        def enter_new_token():
            prompt_text = "Change HuggingFace token:" if saved_token else "Enter HuggingFace token:"
            new_token = tk.simpledialog.askstring("Token Management", prompt_text,
                                                  parent=token_window)
            if new_token:
                save_token(new_token)
                hf_token_var.set("●●●●●●●●" + new_token[-4:])
                token_window.destroy()
                open_token_window()  # Refresh window

        def clear_saved_token():
            if messagebox.askyesno("Confirm", "Clear saved token?", parent=token_window):
                clear_token()
                hf_token_var.set("")
                token_window.destroy()
                open_token_window()  # Refresh window

        # Button text changes based on whether token exists
        button_text = "Change Token" if saved_token else "Add Token"
        tk.Button(button_frame, text=button_text, command=enter_new_token,
                 bg="#3498db", fg="white", font=("Arial", 10, "bold"),
                 padx=15, pady=8).pack(side=tk.LEFT, padx=(0, 10))

        if saved_token:
            tk.Button(button_frame, text="Clear Token", command=clear_saved_token,
                     bg="#e74c3c", fg="white", font=("Arial", 10),
                     padx=15, pady=8).pack(side=tk.LEFT)

        # Close button
        tk.Button(token_window, text="Close", command=token_window.destroy,
                 bg="#95a5a6", fg="white", font=("Arial", 10),
                 padx=20, pady=8).pack(pady=(0, 20))

        # Center window
        token_window.update_idletasks()
        x = (token_window.winfo_screenwidth() // 2) - (token_window.winfo_width() // 2)
        y = (token_window.winfo_screenheight() // 2) - (token_window.winfo_height() // 2)
        token_window.geometry(f"+{x}+{y}")

    # Info panel for estimates and runtime
    info_panel = tk.Frame(right_frame, bg="#ecf0f1", padx=10, pady=10)
    info_panel.pack(fill=tk.X, pady=(0, 10))

    estimate_label = tk.Label(info_panel, text="Estimated processing time: Not calculated yet",
                             font=("Arial", 10, "bold"), bg="#ecf0f1", fg="#27ae60")
    estimate_label.pack(anchor=tk.W)

    runtime_label = tk.Label(info_panel, text="", font=("Arial", 10, "bold"),
                            bg="#ecf0f1", fg="#3498db")
    runtime_label.pack(anchor=tk.W)

    # Text widget with scrollbar for console output
    console_frame = tk.Frame(right_frame, bg="white")
    console_frame.pack(fill=tk.BOTH, expand=True)

    console_scrollbar = ttk.Scrollbar(console_frame)
    console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    console_text = tk.Text(console_frame, wrap=tk.WORD, font=("Consolas", 9),
                          bg="#2c3e50", fg="#ecf0f1",
                          yscrollcommand=console_scrollbar.set,
                          state=tk.DISABLED, padx=10, pady=10)
    console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    console_scrollbar.config(command=console_text.yview)

    # Progress bar section (between console and GPU info)
    progress_frame = tk.Frame(right_frame, bg="#ecf0f1", padx=10, pady=5)
    progress_frame.pack(fill=tk.X, pady=(5, 0))

    progress_label = tk.Label(progress_frame, text="Ready", font=("Arial", 9),
                             bg="#ecf0f1", fg="#2c3e50")
    progress_label.pack(anchor=tk.W, pady=(0, 2))

    progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
    progress_bar.pack(fill=tk.X, pady=(0, 2))

    progress_details = tk.Label(progress_frame, text="", font=("Arial", 8),
                               bg="#ecf0f1", fg="#7f8c8d")
    progress_details.pack(anchor=tk.W)

    # GPU Info at bottom
    gpu_info_frame = tk.Frame(right_frame, bg="#ecf0f1", padx=10, pady=10)
    gpu_info_frame.pack(fill=tk.X, pady=(10, 0))

    try:
        total_vram, available_vram, gpu_name = get_available_vram()
        if total_vram > 0:
            gpu_text = f"GPU: {gpu_name} | Total VRAM: {total_vram:.1f} GB"
            vram_available_text = f"Available VRAM: {available_vram:.1f} GB"
        else:
            gpu_text = f"GPU: {gpu_name}"
            vram_available_text = ""
    except Exception as e:
        print(f"Warning: GPU detection failed: {e}")
        gpu_text = "GPU: Detection failed"
        vram_available_text = ""

    gpu_label = tk.Label(gpu_info_frame, text=gpu_text, font=("Arial", 9, "bold"),
                        bg="#ecf0f1", fg="#34495e")
    gpu_label.pack(anchor=tk.W)

    if vram_available_text:
        vram_label = tk.Label(gpu_info_frame, text=vram_available_text, font=("Arial", 9),
                            bg="#ecf0f1", fg="#3498db")
        vram_label.pack(anchor=tk.W)

    # Control buttons frame (below console)
    control_buttons_frame = tk.Frame(right_frame, bg="#ecf0f1", pady=10)
    control_buttons_frame.pack(fill=tk.X)

    pause_button = [None]  # Store reference for toggling

    def on_stop():
        """Stop processing completely"""
        if processing_thread[0] and processing_thread[0].is_alive():
            if messagebox.askyesno("Stop Processing", "Are you sure you want to stop the current processing?"):
                stop_event.set()
                write_to_console("\n⛔ STOPPING PROCESS...")

    def on_pause_resume():
        """Toggle pause/resume"""
        if processing_thread[0] and processing_thread[0].is_alive():
            if pause_event.is_set():
                # Currently running, so pause it
                pause_event.clear()
                pause_button[0].config(text="▶️  Resume", bg="#27ae60")
                write_to_console("\n⏸️  PAUSED - Click Resume to continue")
            else:
                # Currently paused, so resume it
                pause_event.set()
                pause_button[0].config(text="⏸️  Pause", bg="#f39c12")
                write_to_console("\n▶️  RESUMING...")

    stop_button = tk.Button(control_buttons_frame, text="⏹️  Stop", command=on_stop,
                           bg="#e74c3c", fg="white", font=("Arial", 10, "bold"),
                           cursor="hand2", padx=15, pady=8, state=tk.DISABLED)
    stop_button.pack(side=tk.LEFT, padx=5)

    pause_button[0] = tk.Button(control_buttons_frame, text="⏸️  Pause", command=on_pause_resume,
                               bg="#f39c12", fg="white", font=("Arial", 10, "bold"),
                               cursor="hand2", padx=15, pady=8, state=tk.DISABLED)
    pause_button[0].pack(side=tk.LEFT, padx=5)

    # Function to write to console
    def write_to_console(message):
        console_text.config(state=tk.NORMAL)
        console_text.insert(tk.END, message + "\n")
        console_text.see(tk.END)
        console_text.config(state=tk.DISABLED)
        root.update_idletasks()

    # Progress bar update functions
    def update_progress(percent, task_name="Processing", details=""):
        """Update the progress bar and labels"""
        try:
            progress_bar['value'] = percent
            progress_label.config(text=task_name)
            progress_details.config(text=details)
            root.update_idletasks()
        except Exception:
            pass

    def reset_progress():
        """Reset progress bar to initial state"""
        try:
            progress_bar['value'] = 0
            progress_label.config(text="Ready")
            progress_details.config(text="")
            root.update_idletasks()
        except Exception:
            pass

    # Runtime tracking
    start_time_var = [None]  # Use list to store mutable value

    def update_runtime():
        if start_time_var[0]:
            import time
            elapsed = time.time() - start_time_var[0]
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            secs = int(elapsed % 60)
            time_str = f"{hours}h {mins}m {secs}s" if hours > 0 else f"{mins}m {secs}s"
            runtime_label.config(text=f"Total runtime: {time_str}")
            root.after(1000, update_runtime)  # Update every second

    # Initial welcome message
    write_to_console("="*60)
    write_to_console("TTRPG SESSION SUMMARIZER")
    write_to_console("="*60)
    write_to_console("Configure your settings on the left, then click 'Start Processing'")
    write_to_console("")

    # Configuration variables
    audio_file_var = tk.StringVar()
    processing_mode_var = tk.StringVar(value="single")
    enable_diarization_var = tk.BooleanVar(value=True)
    hf_token_var = tk.StringVar()
    whisper_model_var = tk.StringVar(value="medium.en")
    ollama_chunk_model_var = tk.StringVar(value="qwen2.5:14b")
    chunk_words_var = tk.IntVar(value=400)
    final_words_var = tk.IntVar(value=1200)
    chunk_context_var = tk.IntVar(value=8192)
    final_context_var = tk.IntVar(value=999999)  # 999999 = unlimited

    # Get available Ollama models
    ollama_models = get_ollama_models()
    model_names = [m[0] for m in ollama_models] if ollama_models else ["qwen2.5:14b"]

    # Check for saved HF token
    saved_token = get_saved_token()
    if saved_token:
        hf_token_var.set("●●●●●●●●" + saved_token[-4:] if len(saved_token) > 4 else "●●●●●●●●")

    y_pos = 10

    # ===== MODE SELECTION =====
    mode_frame = tk.LabelFrame(scrollable, text="  Processing Mode  ", font=("Arial", 12, "bold"),
                              bg="white", fg="#2c3e50", padx=15, pady=15)
    mode_frame.pack(fill=tk.X, pady=10)

    task_mode_var = tk.StringVar(value="audio")  # "audio", "existing_transcript", or "existing_chunks"

    def on_mode_change():
        mode = task_mode_var.get()
        if mode == "audio":
            # Show audio processing sections
            section1.pack(fill=tk.X, pady=10)
            processing_mode_frame.pack(fill=tk.X)
            section2.pack(fill=tk.X, pady=10)  # Campaign Character File
            section3.pack(fill=tk.X, pady=10)  # Speaker Diarization
            section4.pack(fill=tk.X, pady=10)  # Whisper Model
            section5.pack(fill=tk.X, pady=10)  # Ollama Models
            # Show "Create new" option in audio mode
            char_new_radio.pack(anchor=tk.W, after=char_none_radio)
            char_help_label.config(text="Track characters/NPCs across multiple sessions in a campaign:")
            # file_label will be updated by browse_file
        elif mode == "existing_transcript":
            # Transcript mode: Skip transcription, show summarization options
            section1.pack(fill=tk.X, pady=10)  # File selection (for transcript file)
            processing_mode_frame.pack_forget()  # No multi-file mode for transcripts
            section2.pack(fill=tk.X, pady=10)  # Campaign Character File
            section3.pack_forget()  # Hide Speaker Diarization
            section4.pack_forget()  # Hide Whisper Model
            section5.pack(fill=tk.X, pady=10)  # Ollama Models (needed for summarization)
            # Hide "Create new" option (no character extraction from transcript)
            char_new_radio.pack_forget()
            if character_file_mode_var.get() == "new":
                character_file_mode_var.set("none")
            char_help_label.config(text="Reference existing campaign character file (optional):")
        else:
            # JSON mode: Hide audio-specific sections but KEEP campaign file (for reference)
            processing_mode_frame.pack_forget()
            section2.pack(fill=tk.X, pady=10)  # KEEP Campaign Character File (for existing file reference)
            section3.pack_forget()  # Hide Speaker Diarization
            section4.pack_forget()  # Hide Whisper Model
            section5.pack(fill=tk.X, pady=10)  # KEEP Ollama Models (needed for final summary)
            # Hide "Create new" option (no character extraction in JSON mode)
            char_new_radio.pack_forget()
            # If "new" was selected, switch to "none"
            if character_file_mode_var.get() == "new":
                character_file_mode_var.set("none")
            char_help_label.config(text="Reference existing campaign character file (optional - characters already extracted during chunk processing):")
            # file_label will be updated by browse_file

    tk.Radiobutton(mode_frame, text="Process Audio Files",
                  variable=task_mode_var, value="audio",
                  bg="white", font=("Arial", 10), command=on_mode_change).pack(anchor=tk.W, pady=2)

    tk.Radiobutton(mode_frame, text="Resummarize Existing Transcript",
                  variable=task_mode_var, value="existing_transcript",
                  bg="white", font=("Arial", 10), command=on_mode_change).pack(anchor=tk.W, pady=2)

    tk.Label(mode_frame, text="(Use this to re-summarize a transcript without re-running Whisper)",
            bg="white", font=("Arial", 8), fg="#7f8c8d").pack(anchor=tk.W, padx=20, pady=(0,5))

    tk.Radiobutton(mode_frame, text="Generate Final Summary from Existing Chunks",
                  variable=task_mode_var, value="existing_chunks",
                  bg="white", font=("Arial", 10), command=on_mode_change).pack(anchor=tk.W, pady=2)

    tk.Label(mode_frame, text="(Use this if you already have chunk summaries and want to regenerate the final summary)",
            bg="white", font=("Arial", 8), fg="#7f8c8d").pack(anchor=tk.W, padx=20)

    # ===== SECTION 1: File Selection =====
    section1 = tk.LabelFrame(scrollable, text="  1. File Selection  ", font=("Arial", 12, "bold"),
                             bg="white", fg="#2c3e50", padx=15, pady=15)
    section1.pack(fill=tk.X, pady=10)

    audio_duration_var = tk.DoubleVar(value=0)  # Store audio duration in seconds

    def update_chunk_context_recommendation(duration_seconds):
        """
        Calculate and display recommended chunk context size based on audio duration

        Logic:
        - Assume ~150 words per minute of speech
        - Chunks are 4000 characters (not words!) with 200 char overlap
        - Average 6 chars per word (5 letters + space) = ~667 words per chunk
        - Each chunk needs context for: system prompt + previous summary + chunk content
        - Add 30% buffer for safety
        - Round up to nearest 2048
        """
        duration_minutes = duration_seconds / 60

        # Estimate words in audio
        estimated_words = duration_minutes * WORDS_PER_MINUTE

        # Convert to characters (avg 6 chars per word including space)
        estimated_chars = estimated_words * 6

        # Calculate number of chunks with overlap
        # chunk_text() uses: start += (chunk_size - overlap) for each step
        effective_step = DEFAULT_CHUNK_SIZE_WORDS - DEFAULT_CHUNK_OVERLAP_WORDS
        num_chunks = max(1, (estimated_chars - DEFAULT_CHUNK_SIZE_WORDS) // effective_step + 1)

        # Average words per chunk (chunk size in chars / 6)
        avg_chunk_words = DEFAULT_CHUNK_SIZE_WORDS / 6

        # Calculate tokens needed for LAST chunk (worst case)
        # Last chunk has ALL previous short contexts accumulated
        # Each previous chunk creates a ~50-word short context
        previous_summaries = max(0, num_chunks - 1)
        short_context_per_chunk = 50  # create_short_context() targets 50 words
        total_previous_context = previous_summaries * short_context_per_chunk

        # Last chunk needs: system prompt + all previous contexts + current chunk content
        system_prompt_words = 300  # Conservative estimate for full prompt
        tokens_per_chunk = ((system_prompt_words + total_previous_context + avg_chunk_words) / 4) * 1.3  # 30% buffer

        # Round up to nearest CONTEXT_INCREMENT
        recommended_context = int(((tokens_per_chunk // CONTEXT_INCREMENT) + 1) * CONTEXT_INCREMENT)

        # Cap at MAX_CONTEXT_SIZE
        recommended_context = min(recommended_context, MAX_CONTEXT_SIZE)

        # Update context if user hasn't manually changed it
        # (Only auto-update if still at default)
        if chunk_context_var.get() == DEFAULT_CONTEXT_SIZE:
            chunk_context_var.set(recommended_context)

        # Show recommendation in context help text with better formatting
        mins = int(duration_minutes)
        context_help.config(
            text=f"(Recommended: {recommended_context:,} tokens for {mins}min audio, ~{int(num_chunks)} chunks)",
            fg="#27ae60", font=("Arial", 8, "bold")
        )

    def update_time_estimate(*args):
        """Update estimated completion time based on selections"""
        duration = audio_duration_var.get()
        if duration == 0:
            estimate_label.config(text="Estimated processing time: Not calculated yet")
            return

        whisper = whisper_model_var.get()
        enable_diar = enable_diarization_var.get()

        # Whisper processing time (ratio of audio duration)
        # Estimates for RTX 5080 with WAV preprocessing and CUDA optimization
        # .en models are ~5-10% faster due to English-only optimization
        whisper_speed = {
            'tiny': 0.02,      # ~1 second per minute of audio
            'tiny.en': 0.018,  # English-only: slightly faster
            'base': 0.03,      # ~2 seconds per minute
            'base.en': 0.027,  # English-only: slightly faster
            'small': 0.06,     # ~4 seconds per minute
            'small.en': 0.054, # English-only: 5-10% faster
            'medium': 0.10,    # ~6 seconds per minute
            'medium.en': 0.09, # English-only: 5-10% faster
            'large': 0.14      # ~8 seconds per minute (multilingual only)
        }

        whisper_time = (duration / 60) * whisper_speed.get(whisper, 0.14)

        # Diarization time (if enabled) - roughly 0.15x audio duration on GPU
        diarization_time = (duration / 60) * 0.15 if enable_diar else 0

        # Summarization time - depends on transcript length
        # Use same calculation as chunk recommendation
        duration_minutes = duration / 60
        estimated_words = duration_minutes * WORDS_PER_MINUTE
        estimated_chars = estimated_words * 6
        effective_step = DEFAULT_CHUNK_SIZE_WORDS - DEFAULT_CHUNK_OVERLAP_WORDS
        num_chunks = max(1, int((estimated_chars - DEFAULT_CHUNK_SIZE_WORDS) // effective_step + 1))

        # Chunk processing: Ollama summarization per chunk
        # ~30 seconds per chunk on average with qwen2.5:14b
        chunk_time = num_chunks * 0.5

        # Final summary: Combining all chunk summaries
        # Scales with number of chunks (more chunks = longer final summary)
        final_time = min(12, max(5, num_chunks * 0.25))

        total_minutes = whisper_time + diarization_time + chunk_time + final_time

        hours = int(total_minutes // 60)
        mins = int(total_minutes % 60)

        time_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
        estimate_label.config(text=f"Estimated processing time: ~{time_str}")

    def browse_file():
        mode = task_mode_var.get()

        if mode == "audio":
            # Browse for audio files
            filenames = filedialog.askopenfilenames(
                title="Select Audio File(s)",
                filetypes=[
                    ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma *.aac"),
                    ("All Files", "*.*")
                ]
            )
            if filenames:
                # Store all filenames (could be one or multiple)
                audio_file_var.set(";".join(filenames))  # Use semicolon separator

                num_files = len(filenames)

                if num_files == 1:
                    # Single file mode
                    filename = filenames[0]
                    processing_mode_var.set("single")

                    # Get audio duration
                    try:
                        import librosa
                        duration = librosa.get_duration(path=filename)
                        audio_duration_var.set(duration)

                        # Format duration display
                        hours = int(duration // 3600)
                        mins = int((duration % 3600) // 60)
                        secs = int(duration % 60)

                        if hours > 0:
                            duration_str = f"{hours}h {mins}m {secs}s"
                        else:
                            duration_str = f"{mins}m {secs}s"

                        file_label.config(text=f"{Path(filename).name} ({duration_str})")
                        update_time_estimate()
                        update_chunk_context_recommendation(duration)
                    except Exception:
                        audio_duration_var.set(0)
                        file_label.config(text=f"{Path(filename).name} (duration unknown)")
                else:
                    # Multiple files mode
                    processing_mode_var.set("multi")
                    audio_duration_var.set(0)  # Can't calculate total easily
                    file_label.config(text=f"{num_files} files selected (multi-file mode)")
                    update_time_estimate()
        elif mode == "existing_transcript":
            # Browse for transcript text file
            filename = filedialog.askopenfilename(
                title="Select Transcript File",
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                audio_file_var.set(filename)
                processing_mode_var.set("single")
                audio_duration_var.set(0)  # No duration for transcript
                file_label.config(text=f"{Path(filename).name}")
        else:
            # Browse for chunk summary JSON file
            filename = filedialog.askopenfilename(
                title="Select Chunk Summary JSON File",
                filetypes=[
                    ("JSON Files", "*_chunk_summaries.json"),
                    ("All JSON Files", "*.json"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                audio_file_var.set(filename)
                file_label.config(text=f"{Path(filename).name}")

    # Create browse button (text will change based on mode)
    browse_button = tk.Button(section1, text="📁 Browse for Audio File(s)", command=browse_file,
              bg="#3498db", fg="white", font=("Arial", 11, "bold"),
              cursor="hand2", padx=20, pady=10)
    browse_button.pack(pady=5)

    # Update button text when mode changes
    def update_browse_button():
        mode = task_mode_var.get()
        if mode == "audio":
            browse_button.config(text="📁 Browse for Audio File(s)")
        elif mode == "existing_transcript":
            browse_button.config(text="📄 Browse for Transcript File")
        else:
            browse_button.config(text="📁 Browse for Chunk Summary JSON")

    task_mode_var.trace_add('write', lambda *_: update_browse_button())

    file_label = tk.Label(section1, text="No file selected", font=("Arial", 9),
                         bg="white", fg="#7f8c8d")
    file_label.pack()

    # Processing mode (within section1) - only shown for audio mode
    processing_mode_frame = tk.Frame(section1, bg="white")
    processing_mode_frame.pack(fill=tk.X)

    tk.Label(processing_mode_frame, text="", bg="white").pack(pady=5)  # Spacer
    tk.Label(processing_mode_frame, text="Processing Mode:", bg="white", font=("Arial", 10, "bold")).pack(anchor=tk.W)
    tk.Radiobutton(processing_mode_frame, text="Single file with speaker diarization",
                   variable=processing_mode_var, value="single",
                   bg="white", font=("Arial", 10)).pack(anchor=tk.W, pady=2)
    tk.Radiobutton(processing_mode_frame, text="Multiple files (one per speaker - Craig bot mode)",
                   variable=processing_mode_var, value="multi",
                   bg="white", font=("Arial", 10)).pack(anchor=tk.W, pady=2)

    # Callback to show/hide speaker diarization based on processing mode
    def on_processing_mode_change(*_):
        """Hide speaker diarization in multi-file mode (speakers already known from filenames)"""
        mode = processing_mode_var.get()
        if mode == "multi":
            # Multi-file mode: hide speaker diarization (not needed)
            section3.pack_forget()
        else:
            # Single-file mode: show speaker diarization
            section3.pack(fill=tk.X, pady=10)

    processing_mode_var.trace_add('write', on_processing_mode_change)

    # ===== SECTION 2: Campaign Character File =====
    section2 = tk.LabelFrame(scrollable, text="  2. Campaign Character File (Optional)  ", font=("Arial", 12, "bold"),
                             bg="white", fg="#2c3e50", padx=15, pady=15)
    section2.pack(fill=tk.X, pady=10)

    character_file_var = tk.StringVar()
    character_file_mode_var = tk.StringVar(value="none")  # "none", "new", or "existing"

    char_help_label = tk.Label(section2, text="Track characters/NPCs across multiple sessions in a campaign:",
            bg="white", font=("Arial", 9), fg="#7f8c8d")
    char_help_label.pack(anchor=tk.W, pady=(0, 10))

    # Radio buttons for character file mode
    char_mode_frame = tk.Frame(section2, bg="white")
    char_mode_frame.pack(fill=tk.X, pady=(0, 10))

    char_none_radio = tk.Radiobutton(char_mode_frame, text="No campaign file (single session only)",
                  variable=character_file_mode_var, value="none",
                  bg="white", font=("Arial", 10))
    char_none_radio.pack(anchor=tk.W)

    char_new_radio = tk.Radiobutton(char_mode_frame, text="Create new campaign character file",
                  variable=character_file_mode_var, value="new",
                  bg="white", font=("Arial", 10))
    char_new_radio.pack(anchor=tk.W)

    char_existing_radio = tk.Radiobutton(char_mode_frame, text="Use existing campaign character file",
                  variable=character_file_mode_var, value="existing",
                  bg="white", font=("Arial", 10))
    char_existing_radio.pack(anchor=tk.W)

    # Character file display and browse button
    char_file_frame = tk.Frame(section2, bg="white")
    char_file_frame.pack(fill=tk.X, pady=(5, 0))

    char_file_label = tk.Label(char_file_frame, text="No campaign file selected",
                               bg="white", fg="#7f8c8d", font=("Arial", 9))
    char_file_label.pack(side=tk.LEFT, padx=(0, 10))

    def browse_or_create_character_file():
        mode = character_file_mode_var.get()
        npcs_dir = DEFAULT_NPCS_DIR
        npcs_dir.mkdir(parents=True, exist_ok=True)

        if mode == "new":
            # Ask for campaign name
            campaign_name = simpledialog.askstring(
                "Campaign Name",
                "Enter a name for your campaign:",
                parent=root
            )
            if campaign_name:
                # Sanitize filename
                safe_name = "".join(c for c in campaign_name if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_name = safe_name.replace(' ', '_')
                char_file = npcs_dir / f"{safe_name}_characters.md"

                # Create initial file
                with open(char_file, 'w', encoding='utf-8') as f:
                    f.write(f"{'='*60}\n")
                    f.write(f"CAMPAIGN: {campaign_name}\n")
                    f.write(f"{'='*60}\n\n")
                    f.write("Characters and NPCs will be tracked here across sessions.\n\n")

                character_file_var.set(str(char_file))
                char_file_label.config(text=f"📝 {char_file.name}", fg="#27ae60")
                print(f"✓ Created new campaign file: {char_file}")

        elif mode == "existing":
            # Browse for existing file
            filename = filedialog.askopenfilename(
                title="Select Campaign Character File",
                initialdir=npcs_dir,
                filetypes=[
                    ("Markdown Files", "*.md"),
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                character_file_var.set(filename)
                char_file_label.config(text=f"📂 {Path(filename).name}", fg="#3498db")
                print(f"✓ Selected campaign file: {filename}")

        else:  # "none"
            character_file_var.set("")
            char_file_label.config(text="No campaign file selected", fg="#7f8c8d")

    tk.Button(char_file_frame, text="📁 Browse/Create", command=browse_or_create_character_file,
             bg="#3498db", fg="white", font=("Arial", 9, "bold"),
             cursor="hand2", padx=10, pady=5).pack(side=tk.LEFT)

    # Update button state based on mode selection
    def update_char_file_button(*_):
        mode = character_file_mode_var.get()
        if mode == "none":
            character_file_var.set("")
            char_file_label.config(text="No campaign file selected", fg="#7f8c8d")

    character_file_mode_var.trace_add('write', update_char_file_button)

    # ===== SECTION 3: Speaker Diarization =====
    section3 = tk.LabelFrame(scrollable, text="  3. Speaker Diarization  ", font=("Arial", 12, "bold"),
                             bg="white", fg="#2c3e50", padx=15, pady=15)
    section3.pack(fill=tk.X, pady=10)

    tk.Checkbutton(section3, text="Enable Speaker Diarization",
                  variable=enable_diarization_var,
                  bg="white", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

    info_text = ("Requires HuggingFace token with access to:\n"
                "  • pyannote/speaker-diarization-3.1\n"
                "  • pyannote/segmentation-3.0")
    tk.Label(section3, text=info_text, bg="white", font=("Arial", 9),
            fg="#7f8c8d", justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 10))

    tk.Button(section3, text="🔑 Manage Token", command=lambda: open_token_window(),
             bg="#9b59b6", fg="white", font=("Arial", 10, "bold"),
             cursor="hand2", padx=15, pady=8).pack(anchor=tk.W)

    # ===== SECTION 4: Whisper Model =====
    section4 = tk.LabelFrame(scrollable, text="  4. Whisper Transcription Model  ",
                             font=("Arial", 12, "bold"),
                             bg="white", fg="#2c3e50", padx=15, pady=15)
    section4.pack(fill=tk.X, pady=10)

    whisper_models = [
        ("tiny", "~1 GB VRAM", "Fastest, least accurate"),
        ("tiny.en", "~1 GB VRAM", "Fastest (English-only, slightly better)"),
        ("base", "~1 GB VRAM", "Fast, basic accuracy"),
        ("base.en", "~1 GB VRAM", "Fast (English-only, slightly better)"),
        ("small", "~2 GB VRAM", "Good balance"),
        ("small.en", "~2 GB VRAM", "Good balance (English-only, 5-10% faster)"),
        ("medium", "~5 GB VRAM", "High accuracy"),
        ("medium.en", "~5 GB VRAM", "High accuracy (English-only, 5-10% faster)"),
        ("large", "~10 GB VRAM", "Best accuracy (multilingual only)")
    ]

    for model, vram, desc in whisper_models:
        frame = tk.Frame(section4, bg="white")
        frame.pack(fill=tk.X, pady=2)
        tk.Radiobutton(frame, text=f"{model.capitalize()}", variable=whisper_model_var,
                       value=model, bg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(frame, text=f"{vram} - {desc}", bg="white", fg="#7f8c8d",
                 font=("Arial", 8)).pack(side=tk.LEFT, padx=10)

    # Update time estimate when whisper model or diarization changes
    whisper_model_var.trace_add('write', update_time_estimate)
    enable_diarization_var.trace_add('write', update_time_estimate)
    # Also update when processing mode changes
    processing_mode_var.trace_add('write', update_time_estimate)

    # ===== SECTION 5: Ollama Models =====
    section5 = tk.LabelFrame(scrollable, text="  5. Ollama Summarization Models  ",
                             font=("Arial", 12, "bold"),
                             bg="white", fg="#2c3e50", padx=15, pady=15)
    section5.pack(fill=tk.X, pady=10)

    # Chunk model
    chunk_frame = tk.LabelFrame(section5, text="Chunk Summaries", bg="white", padx=10, pady=10)
    chunk_frame.pack(fill=tk.X, pady=5)
    chunk_frame.columnconfigure(2, weight=1)  # Allow column 2 to expand
    chunk_frame.columnconfigure(3, weight=1)  # Allow column 3 to expand too

    tk.Label(chunk_frame, text="Model:", bg="white", font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W)
    chunk_model_combo = ttk.Combobox(chunk_frame, textvariable=ollama_chunk_model_var,
                                      values=model_names, state="readonly", width=25)
    chunk_model_combo.grid(row=0, column=1, padx=5, sticky=tk.W)

    chunk_vram_label = tk.Label(chunk_frame, text="", bg="white", fg="#7f8c8d", font=("Arial", 8))
    chunk_vram_label.grid(row=0, column=2, padx=5, sticky=tk.W)

    def update_chunk_vram(*args):
        vram = estimate_vram_usage(ollama_chunk_model_var.get(), chunk_context_var.get())
        # Parse estimated VRAM (format: "~X.X GB")
        try:
            vram_str = vram.replace("~", "").replace("GB", "").strip()
            estimated_gb = float(vram_str)
            _, available_gb, _ = get_available_vram()

            # Turn red if estimated exceeds available
            if available_gb > 0 and estimated_gb > available_gb:
                chunk_vram_label.config(text=f"Est. VRAM: {vram}", fg="#e74c3c")  # Red
            else:
                chunk_vram_label.config(text=f"Est. VRAM: {vram}", fg="#7f8c8d")  # Gray
        except Exception:
            # Fallback on error - show estimate without color coding
            chunk_vram_label.config(text=f"Est. VRAM: {vram}", fg="#7f8c8d")

    ollama_chunk_model_var.trace_add('write', update_chunk_vram)
    chunk_context_var.trace_add('write', update_chunk_vram)

    tk.Label(chunk_frame, text="Target words:", bg="white", font=("Arial", 9)).grid(row=1, column=0, sticky=tk.W, pady=5)
    tk.Spinbox(chunk_frame, from_=100, to=1000, increment=50, textvariable=chunk_words_var,
               width=10).grid(row=1, column=1, sticky=tk.W, pady=5)

    tk.Label(chunk_frame, text="Context size:", bg="white", font=("Arial", 9)).grid(row=2, column=0, sticky=tk.W)
    tk.Spinbox(chunk_frame, from_=2048, to=131072, increment=2048,
               textvariable=chunk_context_var, width=10).grid(row=2, column=1, sticky=tk.W)
    context_help = tk.Label(chunk_frame, text="(tokens of memory - higher = more VRAM, better context)",
                           bg="white", font=("Arial", 8), fg="#7f8c8d", wraplength=500, justify=tk.LEFT, anchor="w")
    context_help.grid(row=2, column=2, sticky=tk.EW, padx=5, columnspan=2)

    # Final summary configuration mode
    final_config_mode_var = tk.StringVar(value="popup")  # "popup" or "now"

    config_mode_frame = tk.Frame(section5, bg="white", pady=10)
    config_mode_frame.pack(fill=tk.X)

    tk.Label(config_mode_frame, text="Final Summary Configuration:",
             bg="white", font=("Arial", 9, "bold"), fg="#2c3e50").pack(anchor=tk.W, pady=(0, 5))

    tk.Radiobutton(config_mode_frame, text="Configure after chunk processing (show popup)",
                  variable=final_config_mode_var, value="popup",
                  bg="white", font=("Arial", 9)).pack(anchor=tk.W, padx=20)

    tk.Radiobutton(config_mode_frame, text="Configure now (automated - no popup)",
                  variable=final_config_mode_var, value="now",
                  bg="white", font=("Arial", 9)).pack(anchor=tk.W, padx=20)

    # Expandable final summary settings panel
    final_settings_frame = tk.Frame(section5, bg="#f8f9fa", padx=15, pady=10)

    # Final summary model
    final_model_frame = tk.Frame(final_settings_frame, bg="#f8f9fa")
    final_model_frame.pack(fill=tk.X, pady=5)

    tk.Label(final_model_frame, text="Final Summary Model:", bg="#f8f9fa",
             font=("Arial", 9, "bold")).grid(row=0, column=0, sticky=tk.W, pady=2)

    final_model_var = tk.StringVar(value=ollama_chunk_model_var.get())
    final_model_combo = ttk.Combobox(final_model_frame, textvariable=final_model_var,
                                      values=model_names, state="readonly", width=25)
    final_model_combo.grid(row=0, column=1, padx=5, sticky=tk.W)

    final_vram_label = tk.Label(final_model_frame, text="", bg="#f8f9fa",
                                fg="#7f8c8d", font=("Arial", 8))
    final_vram_label.grid(row=0, column=2, padx=5, sticky=tk.W)

    # Final summary target words
    tk.Label(final_settings_frame, text="Target words:", bg="#f8f9fa",
             font=("Arial", 9)).pack(anchor=tk.W, pady=(10, 2))
    final_words_var = tk.IntVar(value=1200)
    tk.Spinbox(final_settings_frame, from_=500, to=3000, increment=100,
               textvariable=final_words_var, width=10).pack(anchor=tk.W, padx=0)

    # Final summary context size
    tk.Label(final_settings_frame, text="Context size:", bg="#f8f9fa",
             font=("Arial", 9)).pack(anchor=tk.W, pady=(10, 2))

    final_context_frame = tk.Frame(final_settings_frame, bg="#f8f9fa")
    final_context_frame.pack(fill=tk.X)

    final_context_var = tk.IntVar(value=32768)
    final_context_entry_var = tk.StringVar(value="32,768")

    def update_final_context_display():
        ctx = final_context_var.get()
        if ctx >= 999999:
            final_context_entry_var.set("Unlimited")
        else:
            final_context_entry_var.set(f"{ctx:,}")
        update_final_vram()

    def increment_final_context():
        current = final_context_var.get()
        if current >= 999999:
            pass
        elif current >= 131072:
            final_context_var.set(999999)
        else:
            final_context_var.set(current + 2048)
        update_final_context_display()

    def decrement_final_context():
        current = final_context_var.get()
        if current >= 999999:
            final_context_var.set(131072)
        else:
            final_context_var.set(max(current - 2048, 2048))
        update_final_context_display()

    def update_final_vram(*_):
        vram = estimate_vram_usage(final_model_var.get(),
                                   final_context_var.get() if final_context_var.get() < 999999 else 131072)
        try:
            vram_str = vram.replace("~", "").replace("GB", "").strip()
            estimated_gb = float(vram_str)
            _, available_gb, _ = get_available_vram()

            if available_gb > 0 and estimated_gb > available_gb:
                final_vram_label.config(text=f"Est. VRAM: {vram}", fg="#e74c3c")
            else:
                final_vram_label.config(text=f"Est. VRAM: {vram}", fg="#7f8c8d")
        except Exception:
            # Fallback on error - show estimate without color coding
            final_vram_label.config(text=f"Est. VRAM: {vram}", fg="#7f8c8d")

    final_model_var.trace_add('write', update_final_vram)
    final_context_var.trace_add('write', update_final_vram)

    tk.Entry(final_context_frame, textvariable=final_context_entry_var, width=12,
            font=("Arial", 10), state='readonly').pack(side=tk.LEFT, padx=0)
    tk.Button(final_context_frame, text="▲", command=increment_final_context, width=2).pack(side=tk.LEFT)
    tk.Button(final_context_frame, text="▼", command=decrement_final_context, width=2).pack(side=tk.LEFT, padx=(2, 0))

    # Toggle visibility based on mode
    def on_config_mode_change(*_):
        mode = final_config_mode_var.get()
        if mode == "now":
            final_settings_frame.pack(fill=tk.X, pady=10)
            update_final_vram()
        else:
            final_settings_frame.pack_forget()

    final_config_mode_var.trace_add('write', on_config_mode_change)

    # Initialize VRAM labels
    update_chunk_vram()
    update_final_context_display()

    # ===== BUTTONS =====
    button_frame = tk.Frame(main_frame, bg="#ecf0f1", pady=20)
    button_frame.pack(fill=tk.X)

    # Redirect stdout to console widget
    class ConsoleRedirector:
        def __init__(self, text_widget, root_window, progress_callback=None, original_stdout=None):
            self.text_widget = text_widget
            self.root = root_window
            self.encoding = 'utf-8'
            self.errors = 'replace'
            self.last_line_start = None  # Track where the last line started
            self.progress_callback = progress_callback  # Callback to update progress bar
            self.original_stdout = original_stdout  # Store original stdout for dual output

        def write(self, message):
            try:
                if not message or not message.strip():
                    return

                # Handle Unicode characters properly
                if isinstance(message, bytes):
                    message = message.decode('utf-8', errors='replace')
                else:
                    # Ensure it's a proper string and handle any encoding issues
                    message = str(message).encode('utf-8', errors='replace').decode('utf-8', errors='replace')

                self.text_widget.config(state=tk.NORMAL)

                # Check if this is a progress update (should replace last line)
                # Whisper uses format like: 99%|████████| or [00:23:02.21, 21.5kiB/s]
                # Ollama uses: Progress:, tok/s, etc.
                is_progress = any(keyword in message for keyword in [
                    "Progress:", "Elapsed:", "tok/s", "WPM", "ETA:", "%|", "kiB/s", "MiB/s"
                ])

                if is_progress:
                    # Update visual progress bar but DON'T show in console
                    if self.progress_callback:
                        self._update_progress_from_message(message)
                    # Don't write progress bars to console - they're noisy and shown in visual bar
                    return
                else:
                    # Normal message - append to console
                    self.text_widget.insert(tk.END, message)
                    self.last_line_start = None

                    # ALSO print to original stdout (VS Code console) for debugging
                    if self.original_stdout:
                        try:
                            self.original_stdout.write(message)
                            self.original_stdout.flush()
                        except:
                            pass

                self.text_widget.see(tk.END)
                self.text_widget.config(state=tk.DISABLED)
                self.root.update_idletasks()
            except Exception as e:
                # Print errors to original stdout for debugging
                if self.original_stdout:
                    try:
                        self.original_stdout.write(f"ConsoleRedirector error: {e}\n")
                        self.original_stdout.flush()
                    except:
                        pass

        def _update_progress_from_message(self, message):
            """Extract progress info from message and update progress bar"""
            import re
            try:
                # Only extract percentage if it looks like a real progress indicator
                # Whisper format: "99%|████|" or similar with progress bar
                # Ollama format: "Progress: X tokens" or has "tok/s"
                percent_match = re.search(r'(\d+(?:\.\d+)?)%\|', message)  # Must have %| for progress bar

                if not percent_match:
                    # Try other formats that indicate progress
                    if "tok/s" in message or "WPM" in message:
                        # These don't have percentages but indicate progress
                        # Don't update percentage, just update details
                        return
                    else:
                        return

                percent = int(float(percent_match.group(1)))
                # Clamp to 0-100 range
                percent = max(0, min(100, percent))

                # Determine task name
                task_name = "Processing"
                if "kiB/s" in message or "MiB/s" in message:
                    task_name = "Transcribing Audio"
                elif "tok/s" in message:
                    task_name = "Generating Summary"
                elif "Diarization" in message:
                    task_name = "Speaker Diarization"

                # Extract details (time, speed, etc.)
                details = ""
                time_match = re.search(r'\[(\d+:\d+:\d+)', message)
                speed_match = re.search(r'([\d.]+\s*[kM]iB/s)', message)
                if time_match:
                    details += f"Time: {time_match.group(1)}"
                if speed_match:
                    details += f" | Speed: {speed_match.group(1)}" if details else f"Speed: {speed_match.group(1)}"

                self.progress_callback(percent, task_name, details)

            except Exception:
                pass

        def flush(self):
            pass

    processing_thread = [None]  # Store thread reference
    start_button = [None]  # Store button reference
    stop_event = threading.Event()  # Event to signal stop
    pause_event = threading.Event()  # Event to signal pause
    pause_event.set()  # Start unpaused
    multi_session_name = [None]  # Store session name for multi-file mode

    def run_processing_thread():
        """Background processing thread"""
        try:
            # Reset progress bar at start
            reset_progress()

            # Redirect stdout and stderr to console
            # Keep reference to original stdout for dual output (GUI + VS Code console)
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            redirector = ConsoleRedirector(console_text, root, progress_callback=update_progress, original_stdout=old_stdout)
            sys.stdout = redirector
            sys.stderr = redirector

            # Get config
            audio_files_str = audio_file_var.get()
            audio_files_list = audio_files_str.split(";") if ";" in audio_files_str else [audio_files_str]
            processing_mode = processing_mode_var.get()
            enable_diar = enable_diarization_var.get()

            # Get HuggingFace token if needed
            hf_token = None
            if enable_diar:
                hf_token = get_saved_token()
                if not hf_token:
                    print("WARNING: Diarization enabled but no token found. Disabling diarization.")
                    enable_diar = False

            # Determine multi-file mode
            if len(audio_files_list) > 1:
                # Multi-file processing
                print("\n" + "="*60)
                print("MULTI-FILE MODE")
                print("="*60)
                print(f"Processing {len(audio_files_list)} files")
                print("="*60 + "\n")

                file_to_speaker = assign_speakers_to_files(audio_files_list)

                # Get session name from main thread variable (set before thread started)
                session_name = multi_session_name[0]
                if not session_name:
                    session_name = "multi_file_session"
                print(f"Session name: {session_name}\n")

                summarizer = TTRPGSummarizer(
                    whisper_model=whisper_model_var.get(),
                    ollama_model=ollama_chunk_model_var.get(),
                    enable_diarization=False,
                    chunk_target_words=chunk_words_var.get(),
                    final_summary_target_words=final_words_var.get(),
                    chunk_context_size=chunk_context_var.get(),
                    ollama_final_model=final_model_var.get(),
                    final_context_size=final_context_var.get(),
                    stop_event=stop_event,
                    pause_event=pause_event,
                    campaign_character_file=character_file_var.get() if character_file_var.get() else None,
                    preconfigured_final_summary=(final_config_mode_var.get() == "now")
                )

                results = summarizer.process_multiple_audio_files(file_to_speaker, session_name=session_name)

            else:
                # Single file processing
                print("\n" + "="*60)
                print("SINGLE FILE MODE")
                print("="*60 + "\n")

                audio_file = audio_files_list[0]
                print(f"Selected file: {audio_file}")

                summarizer = TTRPGSummarizer(
                    whisper_model=whisper_model_var.get(),
                    ollama_model=ollama_chunk_model_var.get(),
                    enable_diarization=enable_diar,
                    hf_token=hf_token,
                    chunk_target_words=chunk_words_var.get(),
                    final_summary_target_words=final_words_var.get(),
                    chunk_context_size=chunk_context_var.get(),
                    ollama_final_model=final_model_var.get(),
                    final_context_size=final_context_var.get(),
                    stop_event=stop_event,
                    pause_event=pause_event,
                    campaign_character_file=character_file_var.get() if character_file_var.get() else None,
                    preconfigured_final_summary=(final_config_mode_var.get() == "now")
                )

                results = summarizer.process_audio_file(audio_file)

            print("\n" + "="*60)
            print("PROCESSING COMPLETE!")
            print("="*60)
            print(f"\nResults saved successfully!")

            # Update progress bar to show completion
            update_progress(100, "Complete", "Processing finished successfully")

            # Re-enable start button, disable control buttons
            start_button[0].config(state=tk.NORMAL, text="▶️  Start Processing")
            stop_button.config(state=tk.DISABLED)
            pause_button[0].config(state=tk.DISABLED, text="⏸️  Pause", bg="#f39c12")

        except KeyboardInterrupt:
            # User pressed Stop button - models already unloaded by check_control_events
            print("\n" + "="*60)
            print("PROCESSING STOPPED BY USER")
            print("="*60)
            start_button[0].config(state=tk.NORMAL, text="▶️  Start Processing")
            stop_button.config(state=tk.DISABLED)
            pause_button[0].config(state=tk.DISABLED, text="⏸️  Pause", bg="#f39c12")
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            # Unload models on error to free memory
            try:
                if 'summarizer' in locals():
                    summarizer.unload_models()
            except Exception as e:
                print(f"Warning: Could not unload models: {e}")
            start_button[0].config(state=tk.NORMAL, text="▶️  Start Processing")
            stop_button.config(state=tk.DISABLED)
            pause_button[0].config(state=tk.DISABLED, text="⏸️  Pause", bg="#f39c12")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def on_start():
        mode = task_mode_var.get()

        if not audio_file_var.get():
            if mode == "existing_chunks":
                error_msg = "Please select a chunk summary JSON file!"
            elif mode == "existing_transcript":
                error_msg = "Please select a transcript file!"
            else:
                error_msg = "Please select an audio file!"
            write_to_console(f"❌ ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)
            return

        # Check if we're in "existing transcript" mode
        if mode == "existing_transcript":
            # Process transcript in a separate thread
            def process_transcript():
                try:
                    transcript_file = audio_file_var.get()
                    write_to_console(f"✓ Loading transcript from: {Path(transcript_file).name}")

                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        full_text = f.read()

                    write_to_console(f"✓ Loaded transcript: {len(full_text):,} characters\n")

                    # Create summarizer instance
                    summarizer = TTRPGSummarizer(
                        whisper_model="base",
                        ollama_model=ollama_chunk_model_var.get(),
                        enable_diarization=False,
                        chunk_target_words=chunk_words_var.get(),
                        final_summary_target_words=final_words_var.get(),
                        chunk_context_size=chunk_context_var.get(),
                        ollama_final_model=final_model_var.get(),
                        final_context_size=final_context_var.get(),
                        stop_event=stop_event,
                        pause_event=pause_event,
                        skip_whisper_load=True,
                        campaign_character_file=character_file_var.get() if character_file_var.get() else None,
                        preconfigured_final_summary=(final_config_mode_var.get() == "now")
                    )

                    # Get campaign context
                    campaign_context = summarizer.get_campaign_context()

                    # Chunk and summarize
                    chunks = summarizer.chunk_text(full_text)

                    write_to_console(f"\n{'='*60}")
                    write_to_console(f"CHUNK SUMMARIZATION")
                    write_to_console(f"{'='*60}")
                    write_to_console(f"Total chunks: {len(chunks)}\n")

                    chunk_summaries = []
                    progressive_context = ""

                    for i, chunk in enumerate(chunks, 1):
                        summarizer.check_control_events()
                        summary = summarizer.summarize_chunk(chunk, i, len(chunks), context=campaign_context, previous_context=progressive_context)
                        chunk_summaries.append(summary)

                        short_context = summarizer.create_short_context(summary)
                        progressive_context = (progressive_context + "\n" + short_context) if progressive_context else short_context

                        # Limit context
                        context_words = progressive_context.split()
                        if len(context_words) > 500:
                            progressive_context = " ".join(context_words[-500:])

                        write_to_console("  Building context for next chunk... ✓")

                    # Save outputs
                    output_dir = Path(transcript_file).parent
                    session_name = Path(transcript_file).stem.replace("_transcript", "").replace("_formatted", "")

                    chunk_file = output_dir / f"{session_name}_chunk_summaries_readable.txt"
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        for i, summary in enumerate(chunk_summaries, 1):
                            f.write(f"{'='*80}\nCHUNK {i} of {len(chunk_summaries)}\n{'='*80}\n\n{summary}\n\n")

                    write_to_console(f"\n✓ Chunk summaries saved: {chunk_file.name}")

                    # Final summary
                    overall_summary = summarizer.create_overall_summary(
                        chunk_summaries,
                        context=campaign_context,
                        final_context_size=final_context_var.get(),
                        include_characters=False
                    )

                    model_safe = final_model_var.get().replace(':', '_').replace('/', '_')
                    summary_file = output_dir / f"{session_name}_summary_{model_safe}.txt"
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        f.write(overall_summary)

                    write_to_console(f"✓ Final summary saved: {summary_file.name}")
                    write_to_console(f"\n{'='*60}\n✅ PROCESSING COMPLETE!\n{'='*60}")

                    messagebox.showinfo("Success", f"Summary generated!\n\nSaved to:\n{summary_file}")

                except Exception as e:
                    write_to_console(f"\n❌ ERROR: {str(e)}")
                    messagebox.showerror("Error", f"Failed to process transcript:\n{str(e)}")
                    import traceback
                    traceback.print_exc()

            # Start in background thread
            threading.Thread(target=process_transcript, daemon=True).start()
            return

        # Check if we're in "existing chunks" mode
        if mode == "existing_chunks":
            # Load chunk summaries from JSON and generate final summary
            chunk_file = audio_file_var.get()
            write_to_console(f"✓ Loading chunk summaries from: {Path(chunk_file).name}")

            try:
                import json
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_summaries = json.load(f)

                if not isinstance(chunk_summaries, list):
                    raise ValueError("JSON file must contain a list of chunk summaries")

                write_to_console(f"✓ Loaded {len(chunk_summaries)} chunk summaries")
                write_to_console("")

                # Create a minimal summarizer instance just for the popup and final summary
                summarizer = TTRPGSummarizer(
                    whisper_model="base",  # Not used
                    ollama_model="qwen2.5:14b",  # Not used
                    enable_diarization=False,
                    chunk_target_words=400,
                    final_summary_target_words=final_words_var.get(),
                    chunk_context_size=8192,
                    final_context_size=final_context_var.get(),
                    skip_whisper_load=True,  # Don't load Whisper for JSON-only mode
                    campaign_character_file=character_file_var.get() if character_file_var.get() else None
                )

                # Show popup to configure final summary
                # Don't pass default_context_size - let the popup calculate smart default
                config = summarizer.show_final_summary_config(
                    chunk_summaries,
                    default_target_words=final_words_var.get(),
                    default_context_size=None
                )

                if config is None:
                    write_to_console("⚠️ Final summary generation cancelled by user")
                    return

                model, target_words, context_size, include_chars = config
                write_to_console(f"✓ User confirmed settings:")
                write_to_console(f"  Model: {model}")
                write_to_console(f"  Target words: {target_words}")
                write_to_console(f"  Context size: {'Unlimited' if context_size >= 999999 else f'{context_size:,} tokens'}")
                write_to_console(f"  Include characters: {'Yes' if include_chars else 'No'}")
                write_to_console("")

                # Update settings
                summarizer.ollama_final_model = model
                summarizer.final_summary_target_words = target_words

                # Generate final summary
                write_to_console("Generating final summary...")
                overall_summary = summarizer.create_overall_summary(
                    chunk_summaries,
                    context=None,  # No campaign context available
                    final_context_size=context_size,
                    include_characters=include_chars
                )

                # Save the summary in the same directory as the JSON file (include model name)
                output_dir = Path(chunk_file).parent
                output_name = Path(chunk_file).stem.replace("_chunk_summaries", "")
                model_safe = model.replace(':', '_').replace('/', '_')  # Make model name filesystem-safe
                summary_file = output_dir / f"{output_name}_summary_{model_safe}.txt"

                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(overall_summary)

                write_to_console("")
                write_to_console("="*60)
                write_to_console("FINAL SUMMARY COMPLETE!")
                write_to_console("="*60)
                write_to_console(f"Summary saved to: {summary_file}")
                write_to_console("")

                messagebox.showinfo("Success", f"Final summary generated successfully!\n\nSaved to:\n{summary_file}")

            except Exception as e:
                write_to_console(f"\n❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"Failed to generate summary:\n\n{str(e)}")

            return

        # Normal audio processing mode
        if enable_diarization_var.get() and not saved_token and not hf_token_var.get():
            write_to_console("❌ ERROR: Please provide a HuggingFace token for diarization!")
            messagebox.showerror("Error", "Please provide a HuggingFace token for diarization!")
            return

        write_to_console("✓ Configuration validated successfully!")

        # Parse selected files
        audio_files_str = audio_file_var.get()
        audio_files_list = audio_files_str.split(";") if ";" in audio_files_str else [audio_files_str]

        if len(audio_files_list) == 1:
            write_to_console(f"  - Audio file: {Path(audio_files_list[0]).name}")
        else:
            write_to_console(f"  - Audio files: {len(audio_files_list)} files (multi-file mode)")

        write_to_console(f"  - Whisper model: {whisper_model_var.get()}")
        write_to_console(f"  - Ollama chunk model: {ollama_chunk_model_var.get()}")
        write_to_console(f"  - Diarization: {'Enabled' if enable_diarization_var.get() else 'Disabled'}")
        write_to_console("")
        write_to_console("Starting processing...")

        # Start runtime timer
        import time
        start_time_var[0] = time.time()
        update_runtime()

        # If multi-file mode, ask for session name NOW (before thread starts)
        audio_files_str = audio_file_var.get()
        audio_files_list = audio_files_str.split(";") if ";" in audio_files_str else [audio_files_str]

        if len(audio_files_list) > 1:
            # Multi-file mode - ask for session name
            session_name = simpledialog.askstring(
                "Session Name",
                "Enter a name for this session:",
                initialvalue="multi_file_session",
                parent=root
            )
            if not session_name:
                session_name = "multi_file_session"
            # Sanitize session name for filesystem
            session_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in session_name)
            session_name = session_name.replace(' ', '_')
            multi_session_name[0] = session_name
        else:
            multi_session_name[0] = None

        # Disable start button during processing
        start_button[0].config(state=tk.DISABLED, text="⏳ Processing...")

        # Enable control buttons
        stop_button.config(state=tk.NORMAL)
        pause_button[0].config(state=tk.NORMAL)

        # Reset events
        stop_event.clear()
        pause_event.set()

        # Start processing in background thread
        processing_thread[0] = threading.Thread(target=run_processing_thread, daemon=True)
        processing_thread[0].start()

    def cleanup_and_close():
        """Clean up all models and close the application"""
        write_to_console("\n🧹 Cleaning up models from VRAM...")

        try:
            # Clear PyTorch cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                write_to_console("✓ PyTorch VRAM cache cleared")
        except Exception as e:
            write_to_console(f"⚠️  Could not clear PyTorch cache: {e}")

        try:
            # Get list of loaded Ollama models and unload them
            import subprocess
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # More than just header
                    write_to_console("Unloading Ollama models...")
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            model_name = line.split()[0]
                            try:
                                subprocess.run(['ollama', 'stop', model_name], timeout=5)
                                write_to_console(f"✓ Unloaded: {model_name}")
                            except subprocess.TimeoutExpired:
                                write_to_console(f"⚠️  Timeout unloading: {model_name}")
        except Exception as e:
            write_to_console(f"⚠️  Could not unload Ollama models: {e}")

        write_to_console("✓ Cleanup complete")
        root.after(500, root.destroy)  # Give time to show cleanup messages

    def on_cancel():
        if processing_thread[0] and processing_thread[0].is_alive():
            if messagebox.askyesno("Confirm", "Processing is running. Are you sure you want to cancel?"):
                cleanup_and_close()
        else:
            cleanup_and_close()

    start_button[0] = tk.Button(button_frame, text="▶️  Start Processing", command=on_start,
                                bg="#27ae60", fg="white", font=("Arial", 14, "bold"),
                                cursor="hand2", padx=40, pady=15)
    start_button[0].pack(side=tk.LEFT, padx=10)

    tk.Button(button_frame, text="Close", command=on_cancel,
              bg="#95a5a6", fg="white", font=("Arial", 11),
              cursor="hand2", padx=20, pady=10).pack(side=tk.RIGHT)

    root.mainloop()


def main():
    """Main program with GUI"""
    print("="*60)
    print("TTRPG SESSION SUMMARIZER - Python 3.11 GPU Optimized")
    print("="*60)

    # Check if Ollama is running
    print("\nChecking Ollama availability...")
    while not check_ollama_running():
        print("Ollama is not running!")
        if not prompt_start_ollama():
            print("Exiting - Ollama is required for summarization.")
            return
        print("Checking again...")

    print("Ollama is running!")

    # Show configuration GUI (now handles all processing internally)
    print("\nOpening configuration window...")
    configuration_gui()

    # GUI is closed - processing completed or cancelled
    print("\nProgram exited.")


if __name__ == "__main__":
    main()
