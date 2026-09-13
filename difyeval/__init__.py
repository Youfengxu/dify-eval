"""difyeval — universal Dify eval pipeline.

Dataset (samples) x Task (system under test) x composable Checks
-> Scores -> Metrics -> Experiment.

Built for court-citable determinism: offline byte-stable replays, an
advisory-only LLM judge panel with persisted auditable outputs, and
degrade-never-raise handling of data problems.
"""

__version__ = "0.1.0"
