import os
from dotenv import load_dotenv
from google import genai

# Load .env file so GEMINI_API_KEY is available via os.environ.get() below.
load_dotenv()

# The Gemini model used by every agent in this project.
# Stored here as a constant so all agents use the same model.
# To switch models for the whole project, you change ONE line here.
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Try to read the API key from an environment variable first (the secure way).
# If not set, fall back to the hardcoded key for local development.
# IMPORTANT: before deploying to Streamlit Cloud, put the key in .env
# and NEVER push a hardcoded key to a public GitHub repo.
_api_key = os.environ.get("GEMINI_API_KEY")

# One shared Gemini client imported by every agent that needs AI.
# Creating it once here and reusing it is more efficient than every agent
# creating its own connection on every single call.
gemini_client = genai.Client(api_key=_api_key)
