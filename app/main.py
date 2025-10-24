from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import api_router
from app.core.config import get_settings
from app.core.logging import configure_json_logger, get_logger
from app.storage.sqlite import init_db


configure_json_logger()
log = get_logger("app.main")
settings = get_settings()


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Init SQLite schema
    try:
        await init_db()
        log.info("Initialized SQLite schema")
    except Exception as e:
        log.warning("DB init skipped: {}", e)

    # Lazy-load graph and features into memory caches
    try:
        from app.api.v1 import _ensure_graph  # warm module-level cache

        G = _ensure_graph()
        log.info("Graph cache: nodes={} edges={}", G.number_of_nodes(), G.number_of_edges())
    except Exception as e:
        log.warning("Graph cache not loaded: {}", e)

    try:
        from app.graph.features import load_node_features

        feat_path = Path(settings.model_dir) / "features.parquet"
        if feat_path.exists():
            X = load_node_features(feat_path)
            log.info("Features cache: rows={} cols={}", X.shape[0], X.shape[1])
    except Exception as e:
        log.warning("Features cache not loaded: {}", e)

    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

# CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


def mount_ui_if_present(application: FastAPI) -> None:
    dist_path = Path("ui/web/dist")
    if dist_path.exists():
        log.info("Mounting static UI from {}", dist_path)
        application.mount("/", StaticFiles(directory=str(dist_path), html=True), name="ui")


@app.get("/")
def root():
    dist_index = Path("ui/web/dist/index.html")
    if dist_index.exists():
        return FileResponse(str(dist_index))
    return JSONResponse({
        "message": "UI not built. Run npm install && npm run build in ui/web.",
        "docs": "/docs",
        "api": "/api/v1/health"
    })


mount_ui_if_present(app)
