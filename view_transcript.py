"""
View TTRPG Session Transcripts with Speaker Labels
Displays the formatted transcript with speaker identification
"""

import os
from pathlib import Path
import json
from datetime import datetime


SUMMARIES_DIR = Path(__file__).parent / "summaries"


def list_transcripts():
    """List all available transcripts"""
    transcripts = []

    # Check summaries folder
    if SUMMARIES_DIR.exists():
        for file in SUMMARIES_DIR.glob("*_transcript_formatted.txt"):
            transcripts.append(file)

    # Check main directory for current run
    main_dir = Path(__file__).parent
    for file in main_dir.glob("*_transcript_formatted.txt"):
        transcripts.append(file)

    return sorted(transcripts, key=lambda x: x.stat().st_mtime, reverse=True)


def display_transcript(transcript_file: Path, max_lines: int = None):
    """Display a transcript with formatting"""
    base_name = transcript_file.stem.replace("_transcript_formatted", "")

    print("\n" + "="*80)
    print(f"TTRPG SESSION TRANSCRIPT: {base_name}")
    print("="*80)

    # Show creation date
    mtime = datetime.fromtimestamp(transcript_file.stat().st_mtime)
    print(f"Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for speaker mapping
    speaker_file = transcript_file.parent / f"{base_name}_speakers.json"
    if speaker_file.exists():
        with open(speaker_file, 'r', encoding='utf-8') as f:
            speakers = json.load(f)
        print(f"Speakers detected: {', '.join(speakers.values())}")
    else:
        print("No speaker diarization (transcript only)")

    print("="*80 + "\n")

    # Display transcript
    with open(transcript_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if max_lines:
        print(f"Showing first {max_lines} lines (total: {len(lines)} lines)\n")
        for line in lines[:max_lines]:
            print(line, end='')
        if len(lines) > max_lines:
            print(f"\n\n... ({len(lines) - max_lines} more lines) ...")
    else:
        for line in lines:
            print(line, end='')

    print("\n" + "="*80)
    print(f"File location: {transcript_file}")
    print("="*80)


def search_transcript(transcript_file: Path, search_term: str):
    """Search for a specific term in transcript and show context"""
    base_name = transcript_file.stem.replace("_transcript_formatted", "")

    with open(transcript_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all occurrences
    lines = content.split('\n')
    matches = []

    for i, line in enumerate(lines):
        if search_term.lower() in line.lower():
            # Get context (2 lines before and after)
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            context = '\n'.join(lines[start:end])
            matches.append((i + 1, context))

    if not matches:
        print(f"\nNo matches found for '{search_term}'")
        return

    print(f"\nFound {len(matches)} occurrence(s) of '{search_term}':\n")
    print("="*80)

    for line_num, context in matches:
        print(f"\nLine {line_num}:")
        print("-"*80)
        print(context)
        print("-"*80)


def main():
    """Main viewer"""
    transcripts = list_transcripts()

    if not transcripts:
        print("No transcripts found!")
        print(f"Expected location: {SUMMARIES_DIR}")
        return

    print("="*80)
    print("TTRPG SESSION TRANSCRIPTS")
    print("="*80)
    print(f"Found {len(transcripts)} session(s):\n")

    for i, transcript_file in enumerate(transcripts, 1):
        base_name = transcript_file.stem.replace("_transcript_formatted", "")
        mtime = datetime.fromtimestamp(transcript_file.stat().st_mtime)

        # Check if has speaker info
        speaker_file = transcript_file.parent / f"{base_name}_speakers.json"
        has_speakers = "with speakers" if speaker_file.exists() else "transcript only"

        print(f"{i}. {base_name} ({has_speakers})")
        print(f"   Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Location: {transcript_file.parent.name}/")
        print()

    print("="*80)

    # Ask which to view
    try:
        choice = input("\nEnter number to view (or 'q' to quit): ").strip()

        if choice.lower() == 'q':
            return

        choice_num = int(choice)
        if 1 <= choice_num <= len(transcripts):
            selected = transcripts[choice_num - 1]

            # Ask for display options
            print("\nDisplay options:")
            print("1. Show first 50 lines")
            print("2. Show first 100 lines")
            print("3. Show entire transcript")
            print("4. Search for specific text")

            option = input("\nChoose option (1-4): ").strip()

            if option == '1':
                display_transcript(selected, max_lines=50)
            elif option == '2':
                display_transcript(selected, max_lines=100)
            elif option == '3':
                display_transcript(selected)
            elif option == '4':
                search_term = input("\nEnter search term: ").strip()
                if search_term:
                    search_transcript(selected, search_term)
            else:
                print("Invalid option!")
        else:
            print("Invalid choice!")
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")


if __name__ == "__main__":
    main()
