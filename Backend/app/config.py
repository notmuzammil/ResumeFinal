import os

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "z-ai/glm-5.3-flash").strip()
# Used automatically if the main model is retired, unavailable or times out
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()
# How long "thinking" models reason before answering: low / medium / high, or empty to use the model default
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low").strip()
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "3"))

MAX_RESUMES = int(os.getenv("MAX_RESUMES", "20"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_MB", "10")) * 1024 * 1024

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]


def api_key_configured() -> bool:
    return bool(LLM_API_KEY) and "xxxx" not in LLM_API_KEY
