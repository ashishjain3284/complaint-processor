"""
config.py
---------
All the settings for the project live in this one file.

If you want to change something (the model, the folders, the file types),
change it here. Nothing else in the project has hard-coded settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file (where your OpenAI key is kept) into the environment.
load_dotenv()

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# BASE_DIR is the folder this file sits in, so the paths below work no matter
# which directory you run the program from.
BASE_DIR = Path(__file__).parent

INPUT_FOLDER = BASE_DIR / "data"      # where the complaint documents are read from
OUTPUT_FOLDER = BASE_DIR / "output"   # where all the results are written

# ---------------------------------------------------------------------------
# OpenAI settings
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# gpt-4o-mini is cheap and good enough for this task.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Low temperature = consistent, factual answers (what we want for extraction).
TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# Which document types we can read
# ---------------------------------------------------------------------------
SUPPORTED_FILE_TYPES = [".txt", ".pdf", ".docx"]

# Very long documents are cut to this many characters before being sent to the
# LLM. This keeps the cost predictable.
MAX_CHARACTERS = 15000
