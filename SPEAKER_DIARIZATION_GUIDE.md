# Speaker Diarization Feature Guide

## Overview

Speaker diarization has been added to the TTRPG Summarizer! This feature automatically identifies different speakers in your audio recordings and labels who said what in the transcript.

## Setup

### 1. Install Dependencies

```bash
pip install pyannote.audio
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token" or "Create new token"
3. Settings:
   - **Name**: "TTRPG Diarization" (or any name you like)
   - **Type**: **Read** (this is sufficient)
   - **Expiration**: Optional (can set "No expiration")
4. Copy the token (you'll only see it once!)

### 3. Accept Model Terms (IMPORTANT!)

You must accept the terms for these models:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Click "Agree and access repository" on both pages.

### 4. Token Storage (Optional but Recommended!)

The first time you enter your token, you'll be asked if you want to save it. If you choose "Yes":
- Token is saved to: `~/.ttrpg_summarizer/config.json` (in your home directory)
- You won't need to enter it again in future sessions
- You can update or clear it anytime through the dialogs

**Token Management Options:**
- **First use**: Enter token → Asked if you want to save it
- **Subsequent uses**: Auto-loads saved token → Option to use it, replace it, or clear it
- **Security**: Token stored in plain text in your home directory (only you can access it)

## Usage

When you run the program, you'll see new prompts:

### 1. Enable Diarization Dialog
The program will ask if you want to enable speaker diarization.
- Click **Yes** to enable speaker identification
- Click **No** to skip (faster processing, no speaker labels)

### 2. HuggingFace Token Management
If you enable diarization:

**If you have a saved token:**
- "Saved Token Found" dialog appears
- Choose "Yes" to use the saved token
- Choose "No" to manage the token (enter new one or clear it)

**If you don't have a saved token:**
- Enter your HuggingFace token when prompted
- Choose whether to save it for future use

### 3. Speaker Naming
After diarization completes, you'll see dialogs for each detected speaker:
- **SPEAKER_00**: Usually the most frequent speaker (often the DM)
- **SPEAKER_01**: Second most frequent speaker
- etc.

Enter names like:
- "DM"
- "Alice"
- "Bob"
- "Tank McWarrior"

Or press Cancel to keep the default label.

## Output Files

### With Speaker Diarization Enabled:

1. **`*_speakers.json`** - Speaker ID to name mapping
   ```json
   {
     "SPEAKER_00": "DM",
     "SPEAKER_01": "Alice",
     "SPEAKER_02": "Bob"
   }
   ```

2. **`*_transcript_formatted.txt`** - Enhanced with speaker labels
   ```
   [45.2s - 52.1s] DM:
   You enter the dark tavern and see a hooded figure in the corner.

   [52.5s - 58.3s] Alice:
   I want to approach the figure carefully.
   ```

3. All other files remain the same (raw transcript, summaries, etc.)

## How It Works

### Processing Pipeline:

1. **Speaker Diarization** (2-5 minutes for typical session)
   - Analyzes audio to identify unique voice patterns
   - Creates timeline of who spoke when
   - Detects number of speakers automatically

2. **Speaker Naming** (30 seconds)
   - GUI prompts to assign real names to speaker IDs
   - Saves mapping for reference

3. **Transcription** (5-15 minutes depending on model/length)
   - Whisper converts speech to text
   - Creates timestamped segments

4. **Alignment** (< 1 minute)
   - Matches speaker timeline with transcript segments
   - Assigns speaker label to each piece of text

5. **Summarization** (continues as normal)
   - LLM can now see who said what
   - Creates more contextual summaries

## Technical Details

### GPU Usage:
- Diarization uses ~3-4GB VRAM
- Runs before Whisper (GPU memory is freed after each step)
- Total GPU usage same as before (models run sequentially)

### Accuracy:
- Works best with 2-6 distinct speakers
- Accuracy depends on:
  - Audio quality (clear recording = better results)
  - Speaker overlap (less crosstalk = better results)
  - Voice distinctiveness (different voices = better results)

### Limitations:
- May struggle with very similar voices
- Background music/noise can affect accuracy
- Very short utterances (<1 second) may be misattributed
- Large groups (>8 people) become harder to distinguish

## Troubleshooting

### "HuggingFace token required"
- Make sure you entered the token correctly
- Verify the token has read permissions
- Check that you accepted both model terms

### "Ollama connection error" (unchanged)
- This is unrelated to diarization
- Make sure Ollama is running (`ollama serve`)

### Wrong speaker labels
- The alignment uses segment midpoints (best practice)
- If labels seem swapped, you can manually edit the `*_speakers.json` file
- Rerun with the edited mapping (future feature)

### Out of memory
- Diarization adds ~3-4GB VRAM usage
- If you have <8GB GPU, consider:
  - Using a smaller Whisper model ("base" instead of "large")
  - Disabling diarization for very long files
  - Processing on CPU (slower but uses system RAM)

## Example Output

### Without Diarization:
```
[45.2s - 52.1s]
You enter the dark tavern and see a hooded figure in the corner.
```

### With Diarization:
```
[45.2s - 52.1s] DM:
You enter the dark tavern and see a hooded figure in the corner.

[52.5s - 58.3s] Alice:
I want to approach the figure carefully.

[58.8s - 64.2s] DM:
Roll a stealth check.
```

Much easier to follow who said what!

## Performance Impact

- **Additional Processing Time**:
  - Small sessions (<1 hour): +2-5 minutes
  - Medium sessions (1-3 hours): +5-10 minutes
  - Long sessions (3+ hours): +10-20 minutes

- **GPU Memory**: +3-4GB during diarization phase

- **Disk Space**: +1-2KB for speaker mapping JSON

## Tips for Best Results

1. **Good Audio Quality**:
   - Use a decent microphone
   - Minimize background noise
   - Reduce echo/reverb

2. **Clear Speaker Separation**:
   - Avoid too much crosstalk/overlap
   - Let people finish speaking before responding

3. **Consistent Voices**:
   - Works best when each player maintains consistent voice
   - Character voices may confuse the system

4. **Name Mapping**:
   - Be consistent with names across sessions
   - Use short, clear names ("DM" not "Dungeon Master Bob")

## Future Enhancements (Possible)

- Save token for future sessions (encrypted storage)
- Reuse speaker mapping from previous sessions
- Per-speaker statistics (word count, speaking time)
- Character-specific summaries
- Speaker-aware summarization prompts

## Questions?

Check the main README.md or the code comments in `ttrpg_summarizer.py`.

Enjoy your speaker-labeled transcripts!
