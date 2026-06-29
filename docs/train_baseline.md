# RAO Training Baseline

- Baseline commit: `be7a3e22bf0dc32eca43372e5443148d7334775f`
- Baseline inference tests: 43 passed
- Local validation interpreter: `/opt/anaconda3/bin/python`
- Python: 3.12.4
- Inference trace schema: 0.1.0
- Training trace schema: 1.0.0
- Primary action profile: Existing Harness
- Faithful DeepDive group size: 8
- Faithful DeepDive root batch size: 16
- Faithful DeepDive max depth: 4
- Faithful DeepDive max steps per node: 25
- Faithful DeepDive delegation lambda: 0
- Faithful DeepDive learning rate: 3e-6
- Faithful staleness limit: 3 batches

Implementation choices:

- CISPO lower clip epsilon: 0.2
- CISPO upper clip epsilon: 0.2
- Adam beta1: 0.9
- Adam beta2: 0.95
- Weight decay: 0
- Gradient clip norm: 1
- DeepDive split seed: 42
- Failed verifier-scored trajectories remain trainable
- Harness-generated fallback trajectories are not trainable

Dataset hashes:

- `deepdive_qa_rl.csv`: `126047370f723bd0d665a0ffbe4ff285069a3cabb220afce5657665c02eed2aa`
- `deepdive_qa_sft.csv`: `8912e71b8be8553e2a568b46225064e34d72f07374380c84739d0c17523d5cd3`
- `deepdive_trajectories_sft.csv`: `14f5d020f1b3148789a7c77d3822c45ec9997d63267669f86986128298d3b0cd`
