# TTRPG Session Summarizer

A Python tool to automatically transcribe and summarize tabletop RPG sessions using Whisper and Ollama.

## Features

- **Audio-to-Text**: Converts audio recordings (mp3, wav, m4a, etc.) to text using OpenAI's Whisper
- **Intelligent Chunking**: Splits long transcripts into manageable chunks with overlap
- **AI Summarization**: Uses local Ollama models to summarize each chunk
- **Overall Summary**: Creates a cohesive final summary from all chunks
- **TTRPG-Focused**: Prompts designed to capture story events, combat, NPCs, loot, and more

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements_ttrpg.txt
```

### 2. Install Ollama

Download and install Ollama from: https://ollama.ai

Then pull a model:
```bash
ollama pull llama3.2
# or
ollama pull mistral
```

### 3. Install ffmpeg (required for Whisper)

**Windows:** Download from https://ffmpeg.org/download.html and add to PATH

**Mac:** `brew install ffmpeg`

**Linux:** `sudo apt install ffmpeg`

## Usage

### Basic Usage

```python
from ttrpg_summarizer import TTRPGSummarizer

# Initialize
summarizer = TTRPGSummarizer(
    whisper_model="base",      # tiny, base, small, medium, large
    ollama_model="llama3.2"    # your installed Ollama model
)

# Process an audio file
results = summarizer.process_audio_file("session_recording.mp3")

# Access results
print(results['overall_summary'])
```

### Run the Example

Edit the `main()` function in [ttrpg_summarizer.py](ttrpg_summarizer.py) to point to your audio file:

```python
audio_file = "path/to/your/session_recording.mp3"
```

Then run:
```bash
python ttrpg_summarizer.py
```

## Whisper Model Sizes

| Model  | Size  | Speed | Accuracy | RAM Required |
|--------|-------|-------|----------|--------------|
| tiny   | 39M   | Fast  | Lower    | ~1 GB        |
| base   | 74M   | Fast  | Good     | ~1 GB        |
| small  | 244M  | Medium| Better   | ~2 GB        |
| medium | 769M  | Slow  | Great    | ~5 GB        |
| large  | 1550M | Slower| Best     | ~10 GB       |

**Recommendation:** Start with `base` for testing, use `small` or `medium` for production.

## Output Files

When you process `session_recording.mp3`, the tool creates:

```
summaries/
├── session_recording_transcript.txt      # Full transcription
├── session_recording_chunk_summaries.json # Individual chunk summaries
└── session_recording_summary.txt          # Final overall summary
```

## Customization

### Adjust Chunk Size

```python
chunks = summarizer.chunk_text(
    text,
    chunk_size=4000,   # Characters per chunk
    overlap=200        # Overlap between chunks
)
```

### Custom Summarization Prompts

Edit the prompts in `summarize_chunk()` and `create_overall_summary()` methods to focus on specific aspects of your game.

### Different Ollama Models

```python
summarizer = TTRPGSummarizer(ollama_model="mistral")
```

Available models: `llama3.2`, `mistral`, `llama2`, `codellama`, etc.

## API Reference

### TTRPGSummarizer

**`__init__(whisper_model="base", ollama_model="llama3.2")`**
- Initialize the summarizer with specified models

**`transcribe_audio(audio_file: str) -> Dict`**
- Convert audio to text
- Returns: Dictionary with text, segments, language, metadata

**`chunk_text(text: str, chunk_size=4000, overlap=200) -> List[str]`**
- Split text into overlapping chunks
- Returns: List of text chunks

**`summarize_chunk(chunk: str, chunk_number: int) -> str`**
- Summarize a single chunk using Ollama
- Returns: Summary text

**`create_overall_summary(chunk_summaries: List[str]) -> str`**
- Create final summary from all chunks
- Returns: Overall summary

**`process_audio_file(audio_file: str, output_dir="summaries") -> Dict`**
- Complete pipeline from audio to final summary
- Returns: Dictionary with all outputs and file paths

## Next Steps

This completes objectives 2-4. For objective 1 (Discord audio capture), you'll need to:

1. Use a Discord bot with voice channel access
2. Record audio using `discord.py` with voice support
3. Save recordings and pass to this summarizer

## Troubleshooting

**"ffmpeg not found"**: Install ffmpeg and add to your PATH

**"Ollama connection error"**: Make sure Ollama is running (`ollama serve`)

**Out of memory**: Use a smaller Whisper model (try `tiny` or `base`)

**Slow transcription**: Normal for first run; subsequent runs use cached model
