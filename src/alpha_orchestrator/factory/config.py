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
    # 10 simulations in flight x 12 hours x 20 per hour (about 3 minutes each) = 2,400.
    daily_sim_cap: int = 2400
    active_datasets: int = 1  # 你定的: one dataset at a time
    dataset_sim_cap: int = 2000  # a 4,000-field dataset with signal needs ~1,500
    health_sample: int = 40  # families sampled for the health check (update period, coverage)
    survey_per_stratum: int = 2  # families sampled per stratum for the survey
    survey_max_families: int = 100
    round_size: int = 50  # simulations planned per allocation round once signal is found
    refine_structures: int = 3  # structures tried per refinement round of one family
    combo_count: int = 6  # combinations per arm (llm and random) per strong stratum
    slots: int = 6  # upper bound; lowered automatically to what the platform accepts
    children_per_slot: int = 10  # 规则: 10 alphas per multi-simulation
    min_batch_interval_s: float = 20.0  # 论坛: >3 batches / 10 min triggers 429 storms


@dataclass
class Thresholds:
    noise_sigma: float = 0.40  # measured on this ledger: spread of |Sharpe| for no-signal formulas
    screen_line: float = (
        0.8  # survey queueing line (not a pass line): noise reaches it 13% of the time
    )
    sprt_alpha: float = 0.05  # P(call a no-signal dataset "signal")
    sprt_beta: float = 0.20  # P(miss a dataset with signal)
    sprt_extra_hits: float = 0.12  # a dataset worth working has this many more hits than noise
    min_filled_coverage: float = 0.50  # 默认值 (待你确认): coverage after filling one update period
    suspicious_sharpe: float = 3.0  # 你定的: above this the result is checked, never mined
    suspicious_turnover: float = 0.70
    refine_min_gain: float = 0.15  # 默认值 (待你确认): smaller gains are noise between variants
    refine_max_rounds: int = 3  # 默认值 (待你确认)
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
    model: str = "deepseek-flash"  # grouping fields, descriptions
    strong_model: str = "deepseek-v4-pro"  # combinations of fields
    key_file: str = "~/secrets/deepseek.json"
    temperature: float = 0.4
    timeout_s: float = 120.0


@dataclass
class Policy:
    max_submissions_per_day: int = 4  # 规则
    auto_submit: bool = False  # 设计 ③: you confirm every submission
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
