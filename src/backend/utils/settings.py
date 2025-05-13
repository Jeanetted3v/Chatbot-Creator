from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv(), override=True)


class Settings(BaseSettings):
    """Settings for the FastAPI application."""

    model_config = {
        "env_file": find_dotenv(),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }
    CONFIG_DIR: str = "../../config"

    GOOGLE_CREDENTIALS_PATH: str
    GOOGLE_SPREADSHEET_ID: str
    GOOGLE_CLOUD_PROJECT_ID: str
    MONGODB_URI: str
    OPENAI_API_KEY: str
    AZURE_ENDPOINT: str
    AZURE_API_KEY: str
    AZURE_API_VERSION: str
    ANTHROPIC_API_KEY: str
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    WEBSITE: str


SETTINGS = Settings()
