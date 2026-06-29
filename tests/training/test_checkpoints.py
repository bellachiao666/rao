import pytest

torch = pytest.importorskip("torch")

from recursive_agent_training.checkpoints import atomic_torch_save, load_torch_checkpoint


def test_atomic_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(path, {"value": torch.tensor([1, 2, 3]), "step": 4})
    loaded = load_torch_checkpoint(path)
    assert loaded["step"] == 4
    assert loaded["value"].tolist() == [1, 2, 3]
    assert not (tmp_path / "checkpoint.pt.tmp").exists()


def test_checkpoint_must_contain_a_mapping(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save([1, 2, 3], path)

    with pytest.raises(ValueError, match="state mapping"):
        load_torch_checkpoint(path)


def test_checkpoint_required_keys_are_validated(tmp_path):
    path = tmp_path / "incomplete.pt"
    atomic_torch_save(path, {"version": 1})

    with pytest.raises(ValueError, match="optimizer"):
        load_torch_checkpoint(path, required_keys={"optimizer", "version"})
