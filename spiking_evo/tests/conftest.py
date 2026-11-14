from pathlib import Path

import pytest

from spiking_evo.config import config_from_dict

EXPERIMENTS = Path(__file__).parents[1] / "experiments"


@pytest.fixture
def tiny_config():
    """A run small enough for unit tests (well under a second per generation)."""

    def make(**overrides):
        data = {
            "experiment": "tiny",
            "seed": 3,
            "world": {"width": 12, "height": 12, "steps_per_episode": 60},
            "stimuli": [{"type": "food", "count": 12, "regrow_rate": 0.05}],
            "network": {"hidden_max": 6, "hidden_init": 2, "steps_per_tick": 5},
            "genetics": {
                "population": 12,
                "generations": 4,
                "elitism": 2,
                "crossover": 0.3,
                "mutation": {"add_synapse": 0.3, "add_neuron": 0.2},
            },
            "metrics": {"snapshot_every": 2, "checkpoint_every": 2},
        }
        for key, value in overrides.items():
            if isinstance(value, dict):
                data.setdefault(key, {}).update(value)
            else:
                data[key] = value
        return config_from_dict(data)

    return make
