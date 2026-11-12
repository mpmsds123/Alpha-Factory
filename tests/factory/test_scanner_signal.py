"""The scanner's master table is read for signal fields (synthetic names only)."""

from pathlib import Path

from alpha_orchestrator.factory.study import MASTER_FILE, scanner_signal_fields

HEADER = "地区,数据集,阶,字段,sharpe,turnover,公式\n"


def _master(tmp_path: Path, rows: list[str]) -> None:
    (tmp_path / MASTER_FILE).write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8-sig")


def test_master_table_signal_fields(tmp_path: Path) -> None:
    _master(
        tmp_path,
        [
            "USA,ds1,1阶,f_hit,1.9,0.10,x",
            "USA,ds1,1阶,reduce(f_wrapped),-1.7,0.20,x",  # sign does not matter; wrapper kept
            "USA,ds1,1阶,f_weak,1.2,0.10,x",  # below the submission line
            "USA,ds1,1阶,f_fast,2.0,0.90,x",  # turnover above 70%
            "USA,ds2,1阶,f_other,2.0,0.10,x",
        ],
    )
    got = scanner_signal_fields(str(tmp_path), "USA", "ds1")
    assert got == {"f_hit", "reduce", "f_wrapped"}
    assert scanner_signal_fields(str(tmp_path), "USA", "never_scanned") is None


def test_master_table_dataset_without_signal(tmp_path: Path) -> None:
    _master(tmp_path, ["USA,ds1,1阶,f_weak,0.4,0.10,x"])
    assert scanner_signal_fields(str(tmp_path), "USA", "ds1") == set()
