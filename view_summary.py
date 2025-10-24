"""
View TTRPG Session Summaries
Opens and displays summaries in a readable format
"""

import os
from pathlib import Path
import json
from datetime import datetime

SUMMARIES_DIR = Path(__file__).parent / "summaries"


def list_summaries():
    """List all available summaries"""
    if not SUMMARIES_DIR.exists():
        print("No summaries folder found!")
        return []

    summaries = []
    for file in SUMMARIES_DIR.glob("*_summary.txt"):
        summaries.append(file)

    # Also check main directory for current run
    main_dir = Path(__file__).parent
    for file in main_dir.glob("*_summary.txt"):
        summaries.append(file)

    return sorted(summaries, key=lambda x: x.stat().st_mtime, reverse=True)


def display_summary(summary_file: Path):
    """Display a summary with formatting"""
    base_name = summary_file.stem.replace("_summary", "")

    print("\n" + "="*80)
    print(f"TTRPG SESSION SUMMARY: {base_name}")
    print("="*80)

    # Show creation date
    mtime = datetime.fromtimestamp(summary_file.stat().st_mtime)
    print(f"Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for speaker mapping
    speaker_file = summary_file.parent / f"{base_name}_speakers.json"
    if speaker_file.exists():
        with open(speaker_file, 'r', encoding='utf-8') as f:
            speakers = json.load(f)
        print(f"Speakers: {', '.join(speakers.values())}")

    print("="*80 + "\n")

    # Display summary
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = f.read()

    print(summary)
    print("\n" + "="*80)

    # Show available files
    print("\nAvailable files:")
    transcript = summary_file.parent / f"{base_name}_transcript.txt"
    formatted_transcript = summary_file.parent / f"{base_name}_transcript_formatted.txt"
    chunk_summaries = summary_file.parent / f"{base_name}_chunk_summaries.json"

    if transcript.exists():
        print(f"  - Raw transcript: {transcript}")
    if formatted_transcript.exists():
        print(f"  - Formatted transcript: {formatted_transcript}")
    if speaker_file.exists():
        print(f"  - Speaker mapping: {speaker_file}")
    if chunk_summaries.exists():
        print(f"  - Chunk summaries: {chunk_summaries}")
    print(f"  - Summary: {summary_file}")
    print("="*80)


def main():
    """Main viewer"""
    summaries = list_summaries()

    if not summaries:
        print("No summaries found!")
        print(f"Expected location: {SUMMARIES_DIR}")
        return

    print("="*80)
    print("TTRPG SESSION SUMMARIES")
    print("="*80)
    print(f"Found {len(summaries)} session(s):\n")

    for i, summary_file in enumerate(summaries, 1):
        base_name = summary_file.stem.replace("_summary", "")
        mtime = datetime.fromtimestamp(summary_file.stat().st_mtime)
        print(f"{i}. {base_name}")
        print(f"   Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Location: {summary_file.parent.name}/")
        print()

    print("="*80)

    # Ask which to view
    try:
        choice = input("\nEnter number to view (or 'q' to quit): ").strip()

        if choice.lower() == 'q':
            return

        choice_num = int(choice)
        if 1 <= choice_num <= len(summaries):
            display_summary(summaries[choice_num - 1])
        else:
            print("Invalid choice!")
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")


if __name__ == "__main__":
    main()
