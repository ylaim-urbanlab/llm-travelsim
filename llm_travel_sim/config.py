from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import os

# Load .env from project root (two levels up from this file)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on env vars already being set


@dataclass
class Config:
    # H3 grid
    h3_resolution: int = 8
    cbd_center: Tuple[float, float] = (39.7392, -104.9903)  # Denver lat, lon
    ring_radius: int = 5

    # Simulation timesteps
    timestep_minutes: int = 30
    start_hour: int = 6   # 6am
    end_hour: int = 22    # 10pm

    # Clustering
    n_clusters: int = 6
    n_stories_per_cluster: int = 3

    # LLM
    llm_model: str = "gpt-5.4-mini"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.9
    llm_max_retries: int = 3
    llm_concurrency: int = 4  # parallel hex batches

    # Census
    census_api_key: str = field(default_factory=lambda: os.environ.get("CENSUS_API_KEY", ""))
    census_year: int = 2022

    # OpenAI (optional, for future model switching)
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_org_id: str = field(default_factory=lambda: os.environ.get("OPENAI_ORG_ID", ""))

    # Paths
    cache_dir: str = "cache"
    output_dir: str = "output"

    # Random seed
    random_seed: int = 42

    # Initial population per hex (synthetic baseline)
    base_population: int = 500


DEFAULT_CONFIG = Config()
