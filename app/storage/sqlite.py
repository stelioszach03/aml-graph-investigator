from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.storage.models import Base, Run, CaseLog, ScoreSummary, MetricRun


def get_sqlite_engine(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", echo=False, future=True)
    return engine


def get_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ---------------- Async engine/session using SQLITE_URL ---------------- #


def _sqlite_url() -> str:
    return get_settings().sqlite_url


def get_async_engine(url: Optional[str] = None) -> AsyncEngine:
    url = url or _sqlite_url()
    return create_async_engine(url, echo=False, future=True)


def get_async_session_factory(engine: Optional[AsyncEngine] = None) -> async_sessionmaker[AsyncSession]:
    engine = engine or get_async_engine()
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False, autocommit=False)


async def init_db(engine: Optional[AsyncEngine] = None) -> None:
    engine = engine or get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ---------------- Helpers: create_run, log_case, save_summary ---------------- #


async def create_run(model_version: Optional[str] = None, notes: Optional[Dict[str, Any]] = None,
                     engine: Optional[AsyncEngine] = None) -> int:
    engine = engine or get_async_engine()
    Session = get_async_session_factory(engine)
    async with Session() as session:
        run = Run(model_version=model_version, notes=notes or {})
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return int(run.id)


async def log_case(node_id: str, score: float, label_true: Optional[int] = None,
                   inspector_notes: Optional[str] = None, json_payload: Optional[Dict[str, Any]] = None,
                   engine: Optional[AsyncEngine] = None) -> int:
    engine = engine or get_async_engine()
    Session = get_async_session_factory(engine)
    async with Session() as session:
        entry = CaseLog(node_id=str(node_id), score=float(score), label_true=label_true,
                        inspector_notes=inspector_notes, json_payload=json_payload)
        session.add(entry)
        await session.commit()
        await session.refresh(entry)
        return int(entry.id)


async def save_summary(run_id: int, auc_roc: Optional[float] = None, auc_pr: Optional[float] = None,
                       precision_at_100: Optional[float] = None, precision_at_500: Optional[float] = None,
                       time_to_investigate_ms: Optional[int] = None,
                       engine: Optional[AsyncEngine] = None) -> int:
    engine = engine or get_async_engine()
    Session = get_async_session_factory(engine)
    async with Session() as session:
        row = ScoreSummary(
            run_id=int(run_id),
            auc_roc=None if auc_roc is None else float(auc_roc),
            auc_pr=None if auc_pr is None else float(auc_pr),
            precision_at_100=None if precision_at_100 is None else float(precision_at_100),
            precision_at_500=None if precision_at_500 is None else float(precision_at_500),
            time_to_investigate_ms=None if time_to_investigate_ms is None else int(time_to_investigate_ms),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return int(row.id)


async def create_metric_run(dataset_name: Optional[str], metrics: Dict[str, Any], engine: Optional[AsyncEngine] = None) -> int:
    """Persist a training metric run."""
    import time as _time
    engine = engine or get_async_engine()
    Session = get_async_session_factory(engine)
    async with Session() as session:
        row = MetricRun(
            created_at=int(metrics.get("trained_at") or _time.time()),
            dataset_name=dataset_name or None,
            metrics=metrics,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return int(row.id)


async def get_last_metric_run(engine: Optional[AsyncEngine] = None) -> Optional[MetricRun]:
    engine = engine or get_async_engine()
    Session = get_async_session_factory(engine)
    async with Session() as session:
        from sqlalchemy import select, desc
        res = await session.execute(select(MetricRun).order_by(desc(MetricRun.created_at)).limit(1))
        obj = res.scalar_one_or_none()
        if not obj:
            return None
        return {
            "id": obj.id,
            "created_at": obj.created_at,
            "dataset_name": obj.dataset_name,
            "metrics": obj.metrics,
        }


def _ensure_score_run_summary(engine) -> None:
    """Create a summary table for scoring runs if it doesn't exist.

    Schema: score_run_summary(run_id TEXT, created_at INTEGER epoch seconds,
            k INTEGER, n INTEGER, avg_score REAL, max_score REAL, min_score REAL)
    """
    ddl = text(
        """
        CREATE TABLE IF NOT EXISTS score_run_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            k INTEGER NOT NULL,
            n INTEGER NOT NULL,
            avg_score REAL NOT NULL,
            max_score REAL NOT NULL,
            min_score REAL NOT NULL
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(ddl)


def write_score_run_summary(db_path: Path, run_id: str, created_at: int, stats: Dict[int, Tuple[int, float, float, float]]):
    """Persist top-K score stats into SQLite.

    - db_path: path to SQLite file
    - stats: mapping k -> (n, avg, max, min)
    """
    engine = get_sqlite_engine(db_path)
    _ensure_score_run_summary(engine)
    rows = [
        {
            "run_id": run_id,
            "created_at": int(created_at),
            "k": int(k),
            "n": int(v[0]),
            "avg_score": float(v[1]),
            "max_score": float(v[2]),
            "min_score": float(v[3]),
        }
        for k, v in stats.items()
    ]
    ins = text(
        """
        INSERT INTO score_run_summary (run_id, created_at, k, n, avg_score, max_score, min_score)
        VALUES (:run_id, :created_at, :k, :n, :avg_score, :max_score, :min_score)
        """
    )
    with engine.begin() as conn:
        conn.execute(ins, rows)


def sqlite_path_from_url(url: str) -> Path:
    """Extract filesystem path from a sqlite URL like sqlite:///foo.db or sqlite+aiosqlite:///foo.db"""
    if ":///" in url:
        path = url.split(":::", 1)[-1] if ":::" in url else url.split(":///", 1)[-1]
        return Path(path)
    # Fallback: treat as direct file path
    return Path(url)
