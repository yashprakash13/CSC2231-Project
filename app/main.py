import os
import sys

from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the .env file.
load_dotenv()  

# Build paths inside the project like this: BASE_DIR / 'subdir'.
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)