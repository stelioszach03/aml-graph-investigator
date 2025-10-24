from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(128), nullable=True)
    notes = Column(JSON, nullable=True)

    summaries = relationship("ScoreSummary", back_populates="run", cascade="all, delete-orphan")


class CaseLog(Base):
    __tablename__ = "case_logs"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    node_id = Column(String(128), index=True)
    score = Column(Float)
    label_true = Column(Integer, nullable=True)  # 1/0 or NULL if unknown
    inspector_notes = Column(Text, nullable=True)
    json_payload = Column(JSON, nullable=True)


class ScoreSummary(Base):
    __tablename__ = "score_summary"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    run_id = Column(Integer, ForeignKey("runs.id"), index=True)
    auc_roc = Column(Float, nullable=True)
    auc_pr = Column(Float, nullable=True)
    precision_at_100 = Column(Float, nullable=True)
    precision_at_500 = Column(Float, nullable=True)
    time_to_investigate_ms = Column(Integer, nullable=True)

    run = relationship("Run", back_populates="summaries")


class MetricRun(Base):
    __tablename__ = "metric_runs"
    id = Column(Integer, primary_key=True)
    created_at = Column(Integer)  # unix ts
    dataset_name = Column(String(512), nullable=True)
    metrics = Column(JSON, nullable=True)
