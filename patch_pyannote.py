"""
Patch pyannote.audio to work with NumPy 2.0+
Replaces np.NaN with np.nan in the inference.py file
"""

import os
import sys

# Find the pyannote installation directory
try:
    venv_path = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages")
    venv_path = os.path.normpath(venv_path)

    inference_file = os.path.join(venv_path, "pyannote", "audio", "core", "inference.py")

    if not os.path.exists(inference_file):
        print(f"ERROR: Could not find {inference_file}")
        sys.exit(1)

    print(f"Patching: {inference_file}")

    # Read the file
    with open(inference_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if 'np.nan' in content and 'np.NaN' not in content:
        print("File already patched!")
        sys.exit(0)

    # Replace np.NaN with np.nan (case sensitive)
    original_count = content.count('np.NaN')
    content = content.replace('np.NaN', 'np.nan')

    # Write back
    with open(inference_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Successfully patched {original_count} occurrence(s) of np.NaN → np.nan")
    print("You can now run your TTRPG summarizer!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
