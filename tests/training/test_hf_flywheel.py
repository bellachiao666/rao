from types import SimpleNamespace

from recursive_agent_training.hf_flywheel import prepare_trainable_model


def test_prepare_trainable_model_enables_memory_bounded_training():
    class Model:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)
            self.input_grads_enabled = False
            self.checkpointing_kwargs = None

        def enable_input_require_grads(self):
            self.input_grads_enabled = True

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.checkpointing_kwargs = gradient_checkpointing_kwargs

    model = prepare_trainable_model(Model())
    assert model.config.use_cache is False
    assert model.input_grads_enabled is True
    assert model.checkpointing_kwargs == {"use_reentrant": False}
