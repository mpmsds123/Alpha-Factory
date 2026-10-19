from pathlib import Path

from alpha_orchestrator.domain.enums import DataType
from alpha_orchestrator.domain.models import MetadataSnapshot
from alpha_orchestrator.gates.structural import StructuralValidator
from alpha_orchestrator.seeding import seed_from_file

SEED_FILE = Path(__file__).parents[2] / "seeds" / "first_order.txt"


def test_shipped_first_order_seeds_expand_and_pass_structural_validation() -> None:
    metadata = MetadataSnapshot(
        dataset_id="demo-dataset",
        fields={"assets": DataType.MATRIX, "cash_st": DataType.MATRIX},
        operators={
            "group_rank",
            "ts_backfill",
            "ts_rank",
            "ts_zscore",
            "ts_delta",
            "ts_mean",
            "ts_std_dev",
            "ts_av_diff",
            "multiply",
        },
        settings={},
    )

    proposals = seed_from_file(SEED_FILE, metadata, data_type=DataType.MATRIX, settings={})

    assert len(proposals) == 2 * 6
    validator = StructuralValidator()
    for proposal in proposals:
        result = validator.validate(proposal.expression, {}, metadata, DataType.MATRIX)
        assert result.valid, (proposal.expression, result.issues)
