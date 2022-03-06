import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the .env file.
load_dotenv()  

# Build paths inside the project like this: BASE_DIR / 'subdir'.
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
