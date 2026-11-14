import pytest
import yaml

from spiking_evo.config import ConfigError, config_from_dict, dump_config, load_config

from .conftest import EXPERIMENTS


def test_example_config_matches_spec():
    cfg = load_config(EXPERIMENTS / "food_only_baseline.yaml")
    assert cfg.experiment == "food_only_baseline" and cfg.seed == 42
    assert (cfg.world.width, cfg.world.height, cfg.world.steps_per_episode) == (40, 40, 1000)
    assert cfg.stimuli == [{"type": "food", "count": 60, "regrow_rate": 0.01}]
    assert cfg.network.neuron.tau_m == 20.0 and cfg.network.hidden_max == 30
    assert cfg.genetics.population == 200 and cfg.genetics.mutation.add_synapse == 0.02
    assert cfg.fitness.firing_penalty.target_rate_hz == 20.0
    assert cfg.fitness.exploration.weight == 0.0


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"wrold": {}})
    with pytest.raises(ConfigError, match=r"network\.neuron"):
        config_from_dict({"network": {"neuron": {"tau": 3}}})


def test_types_are_checked():
    with pytest.raises(ConfigError, match="integer"):
        config_from_dict({"world": {"width": 4.5}})
    with pytest.raises(ConfigError, match="encoder"):
        config_from_dict({"network": {"encoder": "morse"}})


def test_toml_and_dump_roundtrip(tmp_path):
    p = tmp_path / "exp.toml"
    p.write_text('experiment = "t"\nseed = 9\n[world]\nwidth = 7\n[[stimuli]]\ntype = "food"\n')
    cfg = load_config(p)
    assert cfg.world.width == 7 and cfg.seed == 9
    again = config_from_dict(yaml.safe_load(dump_config(cfg)))
    assert again == cfg
