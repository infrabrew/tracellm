"""TraceLLM server — FastAPI application with inference and training endpoints."""

from __future__ import annotations

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tracellm import __version__
from tracellm.api.routes import router, set_engine, set_trainer, set_rlhf_manager
from tracellm.config import TraceConfig, load_config
from tracellm.inference.engine import InferenceEngine
from tracellm.training.trainer import TrainingManager
from tracellm.training.rlhf import RLHFManager
from tracellm.utils.logging import setup_logging, get_logger


def create_app(config: TraceConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or load_config()
    log = setup_logging(level=config.logging.level, log_file=config.logging.file)
    logger = get_logger("tracellm.server")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"TraceLLM v{__version__} starting on {config.server.host}:{config.server.port}")
        engine = InferenceEngine(config)
        trainer = TrainingManager(config, engine.registry)
        rlhf = RLHFManager(config, engine.registry)
        set_engine(engine)
        set_trainer(trainer)
        set_rlhf_manager(rlhf)
        logger.info("Engine, trainer, and RLHF manager initialized")
        yield
        engine.loader.unload_all()
        engine.kv_cache.clear()
        logger.info("TraceLLM server shut down")

    app = FastAPI(
        title="TraceLLM",
        description="LLM inference, training, and fine-tuning platform",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


def run_server(
    config_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    workers: int | None = None,
) -> None:
    """Start the TraceLLM server."""
    config = load_config(config_path)
    if host:
        config.server.host = host
    if port:
        config.server.port = port
    if workers:
        config.server.workers = workers

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level=config.logging.level,
    )
