"""
EVE - A Growing Digital Organism Prototype.

EVE is not an LLM Agent, not a Planner, and not a desktop automation assistant.
It is a synthetic-first digital organism with:

  - Realtime sensory streams (screen, audio, cursor, keyboard)
  - State slots for structured perception
  - Low-latency reflex policy (deterministic, simple)
  - Safety inhibition gate (synthetic-only, safe by default)
  - Episode logging (JSONL traces)
  - Offline teacher / training loop (cold path)

Architecture:

  Hot Path (must stay simple, deterministic, low latency):
    SyntheticWorld / Capture -> Detector/Encoder -> StateSlots
    -> Policy -> SafetyGate -> MotorStub -> EpisodeLogger

  Cold Path (never blocks hot path):
    TeacherLabel, RewardSignal, ReplayBuffer, ModelRegistry, TrainingStub

The control_center subpackage is a read-only inspection dashboard for the
hot path — it does NOT control or modify the runtime.
"""

__version__ = "0.5.0"
