import os

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://...")
    REDIS_URL = os.getenv("REDIS_URL", "redis://...")
    PROMETHEUS_PORT = 8001
    ...

settings = Settings()