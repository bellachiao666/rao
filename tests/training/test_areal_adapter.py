import pytest

from recursive_agent_training.optimizers.areal_optimizer import AReaLOptimizerAdapter


def test_areal_adapter_reports_missing_backend():
    with pytest.raises(RuntimeError, match="AReaL"):
        AReaLOptimizerAdapter.from_environment()
