"""Every tunable number in one place, each with where it came from.

Sources: 规则手册 = your Obsidian rules handbook (official docs); API课 = official API
training 2026-04-29; AI课 = official "零基础上手BRAIN-AI篇"; 论坛 = Chinese forum posts;
历史 = your 21,368 IQC simulations.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The repository root (this file is src/alpha_orchestrator/factory/config.py), so the
# command works from any folder.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "factory.yaml"


@dataclass
class Budget:
    daily_sim_cap: int = 4500  # 规则: 5,000/day for consultants; 500 kept for manual work
    explore_fraction: float = 0.30  # 设计 ⑤: month one 30%, later 10-15%
    probe_budget: int = 50  # 官方: "先跑 10-50 次试探"; kill a dataset after this
    slots: int = 6  # 规则: 8 concurrent multi-sims; 2 left for manual use
    children_per_slot: int = 10  # 规则: 10 alphas per multi-simulation
    min_batch_interval_s: float = 20.0  # 论坛: >3 batches / 10 min triggers 429 storms
    max_cells_per_day: int = 12


@dataclass
class Thresholds:
    probe_hit_sharpe: float = 1.0  # 成长经历帖: no |sharpe|>1 in a first-order scan = no signal
    flip_sharpe: float = -1.2  # cnhkmcp 优化手册: keep strong negatives, flip them
    flip_fitness: float = -0.5
    expand_sharpe: float = 1.2  # machine_lib: survivors entering the next order
    expand_fitness: float = 0.7
    finetune_sharpe: float = 1.4  # 优化手册: pure parameter tuning only above this
    finetune_fitness: float = 0.9
    submit_sharpe: float = 1.58  # 规则手册: consultant D1
    submit_sharpe_chn: float = 2.08
    submit_fitness: float = 1.0
    turnover_min: float = 0.01
    turnover_max: float = 0.70
    robust_keep_ratio: float = 0.5  # 研究小组: rank/sign variants keep >=50% of sharpe
    pnl_corr_max: float = 0.7  # 规则: self/prod correlation < 0.7
    dataset_cap: int = 8  # 论坛: <=8-10 submissions per dataset
    margin_bps: dict[str, float] = field(
        default_factory=lambda: {"USA": 5.0, "CHN": 20.0, "IND": 20.0, "MEA": 20.0, "default": 15.0}
    )  # AI课


@dataclass
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"  # routine jobs: expressions, alignment, descriptions
    strong_model: str = "deepseek-v4-pro"  # hypotheses
    key_file: str = "~/secrets/deepseek.json"
    temperature: float = 0.4
    timeout_s: float = 120.0


@dataclass
class Policy:
    max_submissions_per_day: int = 4  # 规则
    auto_submit: bool = False  # 设计 ③: you confirm every submission
    hypotheses_need_approval: bool = True  # 设计 ④
    hypothesis_rounds_before_retire: int = 3
    regions: list[str] = field(default_factory=list)  # empty = every region you can access
    delays: list[int] = field(default_factory=lambda: [1])


@dataclass
class FactoryConfig:
    db_path: str = "~/.alpha_factory/factory.db"
    reports_dir: str = "reports"
    budget: Budget = field(default_factory=Budget)
    thresholds: Thresholds = field(default_factory=Thresholds)
    llm: LLMConfig = field(default_factory=LLMConfig)
    policy: Policy = field(default_factory=Policy)

    @classmethod
    def load(cls, path: Path | None = None) -> "FactoryConfig":
        source = (path or DEFAULT_CONFIG).expanduser()
        if not source.exists():
            return cls(reports_dir=str(REPO_ROOT / cls.reports_dir))
        raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        reports = Path(raw.get("reports_dir", cls.reports_dir)).expanduser()
        return cls(
            db_path=raw.get("db_path", cls.db_path),
            reports_dir=str(reports if reports.is_absolute() else source.parent / reports),
            budget=Budget(**raw.get("budget", {})),
            thresholds=Thresholds(**raw.get("thresholds", {})),
            llm=LLMConfig(**raw.get("llm", {})),
            policy=Policy(**raw.get("policy", {})),
        )

    def margin_floor(self, region: str) -> float:
        table = self.thresholds.margin_bps
        return table.get(region, table["default"])

    def sharpe_floor(self, region: str) -> float:
        t = self.thresholds
        return t.submit_sharpe_chn if region == "CHN" else t.submit_sharpe


def load_llm_key(path: str) -> str:
    """Reads the key at call time; it is never copied into the repo or the ledger."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    key = data.get("api_key") if isinstance(data, dict) else None
    if not key:
        raise ValueError(f'{path} must contain {{"api_key": "..."}}')
    return str(key)
