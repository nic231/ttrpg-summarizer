# Multi-File Mode Guide

The TTRPG Summarizer now supports two processing modes:

## Mode 1: Single File (Original)
**Use when:** You have ONE audio file with multiple speakers mixed together

**Process:**
1. Select single audio file
2. Choose whether to enable speaker diarization
3. If enabled, provide HuggingFace token
4. Diarization identifies "who spoke when"
5. Transcription creates text
6. AI summarizes the session

**Best for:** Traditional recordings from Zoom, OBS, etc.

---

## Mode 2: Multi-File (NEW!)
**Use when:** You have MULTIPLE audio files, one per speaker

**Process:**
1. Select multiple audio files
2. Assign speaker name to each file
3. Each file is transcribed separately with assigned speaker
4. All transcripts are merged chronologically by timestamp
5. AI summarizes the session

**Best for:** Discord Craig bot recordings

### Craig Bot Auto-Detection

The summarizer automatically detects Craig bot filename patterns:
- `username_channelid.wav` → suggests "username"
- `AlexTheGreat_123456789.wav` → suggests "AlexTheGreat"

You can accept the suggestion or type a different name.

---

## How It Works

### Multi-File Processing Flow:

1. **File Selection**
   - Choose all audio files (one per speaker)
   - Works with: MP3, WAV, M4A, FLAC, OGG, WMA, AAC

2. **Speaker Assignment**
   - For each file, you'll see a dialog with:
     - Filename
     - Auto-detected speaker name
   - Press OK to accept or type different name

3. **Transcription**
   - Each file is transcribed separately
   - Speaker label is automatically assigned
   - Progress shown for each file

4. **Merging**
   - All segments are sorted by timestamp
   - Creates chronologically accurate transcript
   - Consecutive segments from same speaker are merged

5. **Summarization**
   - Same AI summarization as single-file mode
   - Campaign context support
   - Detailed chunk summaries + overall summary

---

## Example Usage

### Craig Bot Recording

You downloaded these files from Craig:
```
DM_JohnSmith_123456789.wav
Player1_AliceJones_987654321.wav
Player2_BobWilliams_456789123.wav
Player3_CarolDavis_321654987.wav
```

**Steps:**
1. Run `python ttrpg_summarizer_py311.py`
2. Click "Yes" for multiple files
3. Select all 4 WAV files
4. Assign speakers:
   - File 1: "DM_JohnSmith" → Accept or rename to "John (DM)"
   - File 2: "Player1_AliceJones" → Accept or rename to "Alice"
   - File 3: "Player2_BobWilliams" → Accept or rename to "Bob"
   - File 4: "Player3_CarolDavis" → Accept or rename to "Carol"
5. Processing begins automatically
6. Get merged transcript + summary

---

## Output Files

Both modes create the same output files in `summaries/[basename]/`:

- `*_transcript.txt` - Raw transcript with timestamps
- `*_transcript_formatted.txt` - Human-readable with MM:SS timestamps
- `*_chunk_summaries.json` - Detailed summaries of each section
- `*_summary.txt` - Overall session summary

**Multi-file mode additionally shows:**
- Number of files processed
- List of speakers detected

---

## Performance Notes

### Single File Mode:
- Diarization: 3-5 min (GPU) for 1hr audio
- Transcription: ~15 min (GPU) for 1hr audio
- Summarization: ~10 min for 1hr audio
- **Total: ~30 min**

### Multi-File Mode:
- No diarization needed (already separated!)
- Transcription: ~15 min × number of files (parallel potential)
- Summarization: ~10 min for 1hr audio
- **Total: ~25 min** (slightly faster - no diarization overhead)

---

## Troubleshooting

### "No files selected"
- Make sure you actually select files in the dialog
- Don't cancel the file picker

### "Speaker name is empty"
- You must provide a name for each speaker
- Can't leave the dialog blank

### Transcripts out of sync
- This shouldn't happen - segments are sorted by timestamp
- If it does, check that all audio files start from same time point
- Craig bot recordings should be synchronized automatically

### Wrong speaker names in output
- The names you assign are final
- Make sure to review suggested names before accepting
- You can always re-run with different names

---

## Tips

1. **Naming Speakers:** Use clear, distinguishable names
   - Good: "Alice (Wizard)", "Bob (Fighter)", "Carol (DM)"
   - Bad: "Player1", "Player2", "Speaker3"

2. **File Organization:** Keep Craig recordings together
   - Create folder per session
   - Keep all speaker files in same folder

3. **Campaign Context:** Still works in multi-file mode!
   - You'll be prompted for system/setting/party info
   - Helps AI understand the context

4. **Testing:** Try multi-file mode on a short session first
   - Verify speaker assignments are correct
   - Check transcript merging works properly

---

## Comparison Table

| Feature | Single File | Multi-File |
|---------|-------------|------------|
| **Input** | 1 mixed audio file | Multiple files (1 per speaker) |
| **Diarization** | Required for speaker ID | Not needed |
| **HuggingFace Token** | Required if using diarization | Not required |
| **Setup Time** | Get token, accept terms | Just assign speaker names |
| **Processing Time** | ~30 min for 1hr | ~25 min for 1hr |
| **Accuracy** | Depends on audio quality | Perfect speaker separation |
| **Best For** | Zoom, OBS, general recordings | Discord Craig bot |

---

## Future Enhancements

Potential improvements:
- Parallel transcription of multiple files (would be MUCH faster)
- Better handling of overlapping speech
- Support for video files
- Batch processing multiple sessions

---

**Need help?** Check the main README.md or code comments in [ttrpg_summarizer_py311.py](ttrpg_summarizer_py311.py:1670)
