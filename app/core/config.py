from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    dev = "dev"
    prod = "prod"


class Settings(BaseSettings):
    # Application
    app_name: str = "Graph AML Investigator"
    app_env: AppEnv = AppEnv.dev
    app_version: str = "0.1.0"

    # Storage / Models
    data_dir: str = "./data"
    model_dir: str = "./models/baseline"
    sqlite_url: str = "sqlite+aiosqlite:///./graph_aml.db"

    # Neo4j (optional)
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_pass: Optional[str] = None
    # Compatibility with existing code expecting `neo4j_database` and `neo4j_password`
    neo4j_database: Optional[str] = None

    # API auth (optional simple bearer)
    api_auth_token: Optional[str] = None

    # Graph/Explain params
    graph_page_rank_alpha: float = 0.85
    explain_max_path_len: int = 6
    explain_k_paths: int = 5

    # Load .env if present
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Derived/compatibility properties ---
    @property
    def graph_path(self) -> str:
        # Default graph artifact under data_dir
        return str(Path(self.data_dir) / "interim" / "graph.pkl")

    @property
    def model_path(self) -> str:
        return str(Path(self.model_dir) / "model.pkl")

    @property
    def neo4j_password(self) -> Optional[str]:
        return self.neo4j_pass


@lru_cache()
def get_settings() -> Settings:
    # Singleton settings; reads from .env if present
    return Settings()
