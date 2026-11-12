"""Every number the program decides with, and where it came from.

Program layer: the LLM never sees or changes anything in this file. Sources:
  official docs   the platform's documentation pages, cited by page id and last-modified date
  ledger          measurements on this factory's own earlier runs (date given)
  arithmetic      a calculation shown next to the number
"""

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "factory.yaml"
LLM_KEY_ENV = "ALPHA_FACTORY_LLM_API_KEY"
LOCAL_OVERRIDES = "factory.local.yaml"  # in private_dir; read after factory.yaml


@dataclass
class Gates:
    """Submission criteria, copied from the official docs page "Consultant Submission Tests"
    (id consultant-submission-tests, last modified 2026-09-18). Delay-1 values."""

    source: str = "consultant-submission-tests, 2026-09-18"
    sharpe_min: float = 1.58  # "Greater than 1.58 for Delay 1"
    sharpe_min_chn: float = 2.08  # CHN: "Sharpe >= 2.08, Returns >= 8% and Fitness >= 1.0"
    returns_min_chn: float = 0.08
    fitness_min: float = 1.0  # "Greater than 1 for Delay 1"
    turnover_min: float = 0.01  # "Greater than 1% and less than 70%"
    turnover_max: float = 0.70
    corr_max: float = 0.7  # self- and production-correlation "Less than 0.7"

    def sharpe_line(self, region: str) -> float:
        """The Sharpe line for a region."""
        return self.sharpe_min_chn if region == "CHN" else self.sharpe_min


@dataclass
class Search:
    """How one hypothesis is searched and when it stops."""

    # Official docs "Test Period" (test-period, 2026-09-18): the last N years of the 10-year
    # IS become a test period; selection uses train numbers only, the test period judges
    # the winners. 2 years: the IS-ladder test also starts from the most recent 2 years.
    test_period: str = "P2Y"
    first_round: int = 3  # spec: 1-3 expressions; v7 ideas also started with 3
    # Ledger 2026-10-31: every idea that later reached 1.2 had a first formula >= 1.0.
    # Theory for 3 noise formulas on 8 train years: 95% line = 0.354 x 2.39 = 0.85;
    # measured noise per dataset was wider (sigma 0.36-0.58), so the measured line is used.
    first_gate: float = 1.0
    children: int = 3  # one-change children per improvement round
    max_rounds: int = 3  # 3 + 3 x 3 = 12 simulations at most per hypothesis
    hypothesis_budget: int = 12
    # Ledger: the best of 3 variants correlated 0.7-0.87 beats its parent by 0.34-0.50 from
    # noise alone, so a smaller gain is not evidence of improvement.
    min_gain: float = 0.35
    dry_rounds: int = 2  # two rounds in a row without min_gain end the search
    near_miss: float = 1.2  # one step from the line: keep going up to max_rounds
    # A claim revised after seeing results is the easiest way to overfit: one revision,
    # with half the budget, counted as its own hypothesis.
    max_revisions: int = 1
    revision_budget: int = 6
    impl_retry: int = 3  # extra simulations when round 1 is judged "implementation failed"
    # Owner's decision (v7): above these the result is checked, never mined.
    suspicious_sharpe: float = 3.0
    suspicious_turnover: float = 0.70
    # Out-of-sample gate on the 2-year test period. No-skill Sharpe on 2 years has spread
    # 1/sqrt(2) = 0.71: noise clears 0.5 24% of the time; a real 1.58 alpha that loses 30%
    # out of sample (1.1) clears it about 80% of the time.
    oos_min_sharpe: float = 0.5
    # Per dataset: 30 hypotheses x ~5 sims = ~150 sims (about 6 datasets a day); the forum
    # advises <= 8 submissions per dataset. 10 misses in a row at the round-1 gate: a
    # typical dataset (~20% of hypotheses clear it) does that 0.8^10 = 11% of the time.
    dataset_hypotheses_max: int = 30
    dataset_submissions_max: int = 8
    dataset_dry_streak: int = 10
    # Official docs "Understanding simulation limits": skip ID/date fields, one field per
    # family of similar fields; earlier factory rule (official tip): coverage >= 60%.
    min_field_coverage: float = 0.60
    windows: tuple[int, ...] = (5, 21, 63, 126, 252)  # week, month, quarter, half, year
    # Official docs "Clear these tests": Power Pool <= 8 operators, <= 3 data fields.
    max_operators: int = 8
    max_fields: int = 3


@dataclass
class Budget:
    daily_sim_cap: int = 1000  # this factory's own allocation per quota window
    quota_timezone: str = "America/Los_Angeles"
    quota_reset_hour: int = 21  # the allocation window the scanner also uses
    quota_reserve: int = 0
    slots: int = 6  # upper bound; lowered to what the platform accepts, and remembered
    children_per_slot: int = 10  # platform rule: 10 alphas per multi-simulation
    min_batch_interval_s: float = 20.0  # forum: > 3 batches / 10 min triggers 429 storms
    chunk: int = 20  # simulations between loop decisions
    queue_cap: int = 40  # no new hypothesis while this many simulations wait or run
    active_hypotheses: int = 8  # in flight at once, so the platform stays busy

    def __post_init__(self) -> None:
        if self.daily_sim_cap < 0 or self.quota_reserve < 0:
            raise ValueError("Daily cap and quota reserve must be nonnegative")
        if self.slots < 1 or not 1 <= self.children_per_slot <= 10:
            raise ValueError("Need at least one slot and 1-10 children per batch")
        if not 0 <= self.quota_reset_hour <= 23:
            raise ValueError("Quota reset hour must be 0-23")
        ZoneInfo(self.quota_timezone)

    def window_start(self, now: datetime | None = None) -> datetime:
        """Return the start of the quota window containing now."""
        instant = now or datetime.now(UTC)
        if instant.tzinfo is None:
            raise ValueError("Quota clock must include a timezone")
        local = instant.astimezone(ZoneInfo(self.quota_timezone))
        cut = local.replace(hour=self.quota_reset_hour, minute=0, second=0, microsecond=0)
        if local < cut:
            cut -= timedelta(days=1)
        return cut.astimezone(UTC)


@dataclass
class Cost:
    # Estimate: ~200 hypotheses/day x ~4 calls x ~$0.005 = ~$4, plus headroom.
    daily_usd_cap: float = 5.0
    # Owner's target: LLM dollars per passing alpha. Checked over a trailing window once
    # enough has been spent for the ratio to mean something.
    usd_per_pass_max: float = 1.6
    guard_window_days: int = 7
    guard_min_spend_usd: float = 5.0


@dataclass
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    key_file: str = "~/secrets/deepseek.json"
    # Ideas and their first expressions decide everything downstream: stronger model.
    # Diagnosis and decision are short, structured judgements: cheaper model. Every
    # decision row records the model, so passing alphas per $ can overturn this.
    models: dict[str, str] = field(
        default_factory=lambda: {
            "hypothesis": "deepseek-v4-pro",
            "expression": "deepseek-v4-pro",
            "diagnosis": "deepseek-flash",
            "decision": "deepseek-flash",
            "prior_seed": "deepseek-flash",
        }
    )
    max_tokens: dict[str, int] = field(
        default_factory=lambda: {
            "hypothesis": 1200,
            "expression": 1500,
            "diagnosis": 800,
            "decision": 1000,
            "prior_seed": 1500,
        }
    )
    temperature: float = 0.4
    timeout_s: float = 120.0
    thinking: bool = False  # keeps bounded JSON output from being consumed by reasoning
    # The private language notes are ~33k characters (~15k tokens); sending them whole on
    # every expression call would cost more than the rest of the prompt. First 6,000 only.
    notes_chars: int = 6000
    # Provider's official price page (api-docs.deepseek.com/quick_start/pricing, read
    # 2026-11-12): USD per 1M tokens [input cache miss, output] at PEAK; off-peak is half.
    # Peak hours: 01-04 and 06-10 UTC, Monday-Friday.
    prices_usd_peak: dict[str, list[float]] = field(
        default_factory=lambda: {
            "deepseek-v4-pro": [1.32, 3.96],
            "deepseek-flash": [0.30, 1.20],
        }
    )
    peak_hours_utc: tuple[tuple[int, int], ...] = ((1, 4), (6, 10))


@dataclass
class Memory:
    k_records: int = 5  # experience records shown per decision
    k_priors: int = 5  # priors shown per decision
    # Same kind of data and same kind of idea matter most; field type and region less.
    weights: dict[str, int] = field(
        default_factory=lambda: {
            "category": 3,
            "hypothesis_type": 2,
            "field_type": 1,
            "region": 1,
        }
    )
    min_score: int = 3  # below this a record is not "similar"; nothing is shown
    # Initial Beta(a, b) by source: worth about 4-5 observations, so a handful of the
    # factory's own results can overturn a claim.
    prior_seed: dict[str, list[float]] = field(
        default_factory=lambda: {
            "official": [3.0, 2.0],
            "forum": [2.0, 2.0],
            "own": [1.0, 1.0],
        }
    )
    seed_usd_cap: float = 0.5  # one-time cost cap for extracting priors from documents


@dataclass
class Policy:
    regions: list[str] = field(default_factory=list)  # empty = every region offered
    delay: int = 1  # spec: delay-1 only for now
    max_submissions_per_day: int = 4  # platform rule; submission is always by hand


@dataclass
class FactoryConfig:
    db_path: str = "~/.alpha_factory/factory_v8.db"
    reports_dir: str = "reports"
    private_dir: str = "~/.alpha_factory/private"  # platform vocabulary, never in the repo
    scanner_runs: str = ""  # the template scanner's runs folder, for the originality check
    v7_db: str = "~/.alpha_factory/factory.db"  # read-only source for `memory import-v7`
    # one expiring, digest-pinned permission to resume a quarantined batch (see quarantine.py)
    quarantine_exception: dict[str, str] = field(default_factory=dict)
    gates: Gates = field(default_factory=Gates)
    search: Search = field(default_factory=Search)
    budget: Budget = field(default_factory=Budget)
    cost: Cost = field(default_factory=Cost)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: Memory = field(default_factory=Memory)
    policy: Policy = field(default_factory=Policy)

    @classmethod
    def load(cls, path: Path | None = None) -> "FactoryConfig":
        """Load factory.yaml, falling back to defaults for missing keys. Gates are not
        overridable: they come from the official docs."""
        source = (path or DEFAULT_CONFIG).expanduser()
        raw: dict[str, Any] = {}
        if source.exists():
            raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        # Machine-specific values (personal paths) live next to the private vocabulary.
        local = Path(raw.get("private_dir", cls.private_dir)).expanduser() / LOCAL_OVERRIDES
        if local.is_file():
            for key, value in (yaml.safe_load(local.read_text(encoding="utf-8")) or {}).items():
                if isinstance(value, dict) and isinstance(raw.get(key), dict):
                    raw[key] = {**raw[key], **value}
                else:
                    raw[key] = value
        base = source.parent if source.exists() else REPO_ROOT
        reports = Path(raw.get("reports_dir", cls.reports_dir)).expanduser()
        search = dict(raw.get("search") or {})
        if "windows" in search:
            search["windows"] = tuple(search["windows"])
        llm = dict(raw.get("llm") or {})
        defaults = LLMConfig()
        for key in ("models", "max_tokens", "prices_usd_peak"):
            if key in llm:
                llm[key] = {**getattr(defaults, key), **llm[key]}
        return cls(
            db_path=raw.get("db_path", cls.db_path),
            reports_dir=str(reports if reports.is_absolute() else base / reports),
            private_dir=raw.get("private_dir", cls.private_dir),
            scanner_runs=raw.get("scanner_runs", cls.scanner_runs),
            v7_db=raw.get("v7_db", cls.v7_db),
            quarantine_exception=dict(raw.get("quarantine_exception") or {}),
            search=Search(**search),
            budget=Budget(**(raw.get("budget") or {})),
            cost=Cost(**(raw.get("cost") or {})),
            llm=LLMConfig(**llm),
            memory=Memory(**(raw.get("memory") or {})),
            policy=Policy(**(raw.get("policy") or {})),
        )


def llm_key_available(path: str) -> bool:
    """Return True when an LLM key is set in the environment or the key file exists."""
    return bool(os.environ.get(LLM_KEY_ENV)) or Path(path).expanduser().exists()


def load_llm_key(path: str) -> str:
    """Read the LLM key at call time (environment first, then the key file)."""
    from_env = os.environ.get(LLM_KEY_ENV)
    if from_env:
        return from_env
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    key = data.get("api_key") if isinstance(data, dict) else None
    if not key:
        raise ValueError(f'{path} must contain {{"api_key": "..."}}')
    return str(key)
