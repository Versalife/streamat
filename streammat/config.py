from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    streammat_cache_dir: str = "/tmp/streammat_cache"

settings = Settings()
