# TTRPG Summarizer Development Log

This file tracks all development sessions, decisions, and changes made to the project.

---

## Session: 2025-01-24 (Continuation Session)

### Context
This session continued from a previous conversation that ran out of context. The application is a TTRPG session transcription and summarization tool using:
- OpenAI Whisper (local GPU) for transcription
- PyAnnote Audio for speaker diarization
- Ollama (qwen2.5:14b) for AI summarization
- Campaign character tracking across sessions

### Issues Fixed

#### 1. Temp WAV File Cleanup Error
**Problem:** `FileNotFoundError` - temp WAV file was being deleted before speaker audio clip extraction could use it.

**Root Cause:**
- Diarization creates a temp WAV file
- File was being cleaned up at end of `perform_diarization()` (line 680-683)
- Speaker clip extraction needed this file later

**Solution:**
- Removed premature cleanup from `perform_diarization()`
- Added cleanup after speaker identification completes (line 2714-2723)
- Tracks both transcription and diarization temp WAV files separately

**Files Modified:** `ttrpg_summarizer_py311.py` lines 663-665, 670-672, 680-681, 2714-2723

---

#### 2. Default Whisper Model
**Problem:** Default model was set to `large` which is slow.

**Solution:** Changed default to `medium.en` for better balance of speed/accuracy for English-only audio.

**Files Modified:** `ttrpg_summarizer_py311.py` line 3664

---

#### 3. Transcript Spam in Console
**Problem:** Whisper's `verbose=True` was dumping thousands of transcript lines into the GUI console, causing massive slowdown and freezing.

**Root Cause:** Verbose mode prints every transcribed segment with timestamps like:
```
[01:35:33.580] And who do we have here?
[01:35:35.700] We have a new visitor...
```

**Solution:** Created `WhisperProgressFilter` class that:
- Intercepts stdout during transcription
- Shows progress bars (containing `%|`)
- Hides transcript lines (starting with `[timestamp]`)
- Shows errors/warnings

**Files Modified:** `ttrpg_summarizer_py311.py` lines 427-463

**Why This Approach:** Previous ProgressWriter class was removed, but we still need progress updates without transcript spam. The filter gives us the best of both worlds.

---

#### 4. Git Repository Setup
**User Request:** Set up git repository for better version control and collaboration with Claude.

**Actions:**
1. Created `.gitignore` with Python-specific exclusions (audio files, outputs, cache, etc.)
2. Initialized git repo: `git init`
3. Created initial commit with all main project files
4. Installed GitHub CLI via `winget`
5. Authenticated with GitHub (user did this manually)
6. Created public repo: `ttrpg-summarizer` on user's GitHub account (nic231)
7. Pushed to: https://github.com/nic231/ttrpg-summarizer

**Benefits:**
- Version history for all changes
- Easy rollback if issues arise
- Collaboration between sessions
- Backup in the cloud

**Agreement:** Auto-push commits to GitHub after each significant change.

---

#### 5. VAD (Voice Activity Detection) for Multi-File Mode
**Problem:** In Craig mode (multi-file processing), each speaker has their own track that's ~80% silence. Whisper still processes the full duration, wasting time.

**Solution:**
- Added `use_vad` parameter to `transcribe_audio()` method
- Enabled `vad_filter=True` automatically for multi-file mode
- Single-file mode unchanged (all speakers concurrent, no wasted silence)

**Benefits:**
- 3-4x faster transcription for individual speaker tracks
- Only processes actual speech sections
- No impact on timestamp accuracy

**Files Modified:** `ttrpg_summarizer_py311.py` lines 384-390, 451-465, 2377-2380

**Commit:** `407b940`

---

#### 6. Character Deduplication Issues
**Problem:** Character file had massive duplicates:
- "Jasper" vs "Jasper (Player: Alexander Ward)" vs "Jasper (mentioned through text)"
- "Victor Temple" vs "Victor Temple (via remote support)"
- "Annabelle" vs "Annabelle (Potential NPC)"

AI was adding qualifiers/parentheticals that made names different strings.

**Solution 1 - Name Normalization:**
- Strip parenthetical qualifiers: `(Player: X)`, `(NPC)`, `(mentioned...)`, etc.
- Remove "via X" suffixes
- Use normalized name as dictionary key
- Example: "Jasper (Player: Alexander)" → "Jasper"

**Solution 2 - Updated AI Prompt:**
- Added explicit rules telling AI to use core names only
- Provided examples of what NOT to do
- Strong guidance to avoid qualifiers

**Files Modified:** `ttrpg_summarizer_py311.py` lines 1695-1703 (normalization), 1646-1651 (AI prompt)

**Commit:** `00d9129`

---

#### 7. Campaign File Format
**Problem:**
- Campaign files were created as `.txt` instead of `.md`
- User expected files in `E:\Dropbox\Python\TTRPG Sumariser\NPCs\` directory
- Markdown format better for human editing and GitHub preview

**Solution:**
- Changed extension from `.txt` to `.md`
- Files already created in correct NPCs directory
- Updated file browser to prioritize `.md` files

**Benefits:**
- Better formatting in editors with markdown preview
- Renders nicely on GitHub
- Still editable as plain text

**Files Modified:** `ttrpg_summarizer_py311.py` lines 4037, 4056-4058

**Commit:** `00d9129`

---

#### 8. Session Names Instead of Chunk Numbers
**Problem:** Character first appearance showed "Chunk 3" which is meaningless when reviewing campaign files later. "Episode 1" or "Session 5" is much more useful.

**Solution:**
- Added `current_session_name` instance variable
- Session name derived from audio filename (e.g., "Episode 1")
- Characters now tracked as "Episode 1 (Chunk 3)" instead of just "Chunk 3"
- Updated load/save functions to handle new format
- Works for both single-file and multi-file modes

**Before:**
```
First appeared: Chunk 3
```

**After:**
```
First appeared: Episode 1 (Chunk 3)
```

**Benefits:**
- More meaningful character tracking across campaign
- Easy to see which session a character first appeared in
- Chunk number still preserved for within-session reference

**Files Modified:**
- `ttrpg_summarizer_py311.py` lines 318 (tracking var), 2695 (store session name), 1627 (pass to extraction), 1635-1643 (function signature), 1733-1739 (use session name), 1757-1771 (load format), 1820 (save format), 2617 (session file format), 2987 (multi-file format)

**Commit:** `e36f8b9`

---

### Decisions Made

1. **Auto-push to GitHub:** All commits are automatically pushed to GitHub after creation.

2. **Ask about Git for new projects:** When user starts a new project, ask if they'd like to manage it in their git repositories.

3. **Markdown for campaign files:** Use `.md` extension for better human readability and GitHub integration.

4. **VAD only for multi-file mode:** Single-file mode doesn't benefit from VAD since all speakers are concurrent.

5. **Session names from filename:** Use audio filename as session identifier for character tracking.

---

### Current State

**Git Repository:** https://github.com/nic231/ttrpg-summarizer

**Latest Commits:**
- `e36f8b9` - Track characters by session name instead of chunk number
- `00d9129` - Improve character deduplication and campaign file handling
- `407b940` - Add VAD (Voice Activity Detection) for multi-file mode
- `b043aa1` - Update .gitignore to exclude temp files and old versions
- `77ad85b` - Initial commit: TTRPG Summarizer with Whisper + Ollama

**Key Features:**
- ✅ Audio transcription with Whisper (GPU-accelerated)
- ✅ Speaker diarization with PyAnnote Audio
- ✅ AI summarization with Ollama
- ✅ Campaign character tracking with deduplication
- ✅ Progressive context building for chunk summaries
- ✅ Visual progress tracking GUI
- ✅ English-only Whisper models (.en variants)
- ✅ VAD for multi-file mode efficiency
- ✅ Session-based character tracking
- ✅ Markdown campaign files

**Known Working:**
- Temp WAV cleanup (both transcription and diarization)
- Character deduplication with name normalization
- Session name tracking in character files
- WhisperProgressFilter prevents console spam

---

### Technical Notes

**Whisper Progress Filtering:**
The `WhisperProgressFilter` class intercepts Whisper's verbose output:
- Progress bars contain `%|` pattern → shown in console
- Transcript lines start with `[timestamp]` → filtered out
- Other text (errors, warnings) → shown in console

This prevents GUI freezing from thousands of transcript lines while maintaining progress visibility.

**Character Name Normalization:**
Regex patterns used:
- `r'\s*\([^)]*\)\s*$'` - Removes trailing parenthetical qualifiers
- `r'\s+via\s+.*$'` - Removes "via X" suffixes

Applied before dictionary lookup to ensure variants of same character are merged.

**Session Name Storage:**
Session name stored in `self.current_session_name` and passed through:
1. `process_audio_file()` extracts from filename
2. Stored in instance variable
3. Passed to `extract_characters_from_summary()`
4. Used in character first_appearance field
5. Saved/loaded in campaign files

---

### Future Considerations

1. **Testing:** Current changes haven't been tested on actual audio yet. User is in middle of processing "Episode 1" when these changes were made.

2. **Backward Compatibility:** Old character files with "Chunk X" format are still loadable due to flexible regex pattern.

3. **Multi-session Campaign Files:** When a character appears in multiple sessions, mentions are counted but first_appearance never updates (by design - shows where they first appeared).

---

## End of Session Log

This log will be updated with each development session to maintain context across conversations.
