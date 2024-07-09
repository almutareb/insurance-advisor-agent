import os
from pydantic import AnyHttpUrl, ConfigDict
from pydantic_settings import BaseSettings
from enum import Enum


class ModeEnum(str, Enum):
    development = "development"
    production = "production"
    testing = "testing"


class Settings(BaseSettings):
    PROJECT_NAME: str = "app"
    BACKEND_CORS_ORIGINS: list[str] | list[AnyHttpUrl]
    MODE: ModeEnum = ModeEnum.development
    API_VERSION: str = "v1"
    API_V1_STR: str = f"/api/{API_VERSION}"
    HUGGINGFACEHUB_API_TOKEN: str
    GOOGLE_CSE_ID: str
    GOOGLE_API_KEY: str
    VECTOR_DATABASE_LOCATION: str
    #CONVERSATION_COLLECTION_NAME: str
    EMBEDDING_MODEL: str
    SOURCES_CACHE: str
    #LOCAL_CACHE: str
    LLM_MODEL: str

    class Config:
        case_sensitive = True
        env_file = os.path.expanduser(".env")


settings = Settings()