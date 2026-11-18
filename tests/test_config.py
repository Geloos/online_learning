from drift_ensemble import config


def test_tuning_and_test_seeds_disjoint():
    assert not set(config.TUNING_SEEDS) & set(config.TEST_SEEDS)


def test_mixed_split_matches_n_drifts():
    assert sum(config.MIXED_SPLIT) == config.N_DRIFTS


def test_delay_grid():
    assert config.DELAY_GRID == [50, 100, 150, 200, 250, 300, 350, 400,
                                 450, 500]
    assert config.DELAY_GRID[-1] == config.MAX_DELAY


def test_one_strike_mute():
    # W_MIN is reached after exactly one FALSE_ALARM penalty.
    assert config.W_INIT > config.W_MIN
    assert config.W_INIT * config.BETA <= config.W_MIN


def test_clusters_cover_all_detectors():
    members = [m for ms in config.CLUSTER_MEMBERS.values() for m in ms]
    assert sorted(members) == sorted(config.DETECTOR_ORDER)
    assert set(members) == set(config.TUNING_GRIDS)


def test_verification_fits_warmup():
    # Assert t_a >= N+D is guaranteed by MIN_TRAINING_SIZE.
    for cfg in config.VERIFY.values():
        assert cfg["N"] + cfg["D"] <= config.MIN_TRAINING_SIZE


def test_grid_sizes():
    import math

    sizes = {name: math.prod(len(v) for v in grid.values())
             for name, grid in config.TUNING_GRIDS.items()}
    assert sizes == {"STEPD": 36, "SEED": 700, "HDDMWeighted": 50,
                     "RDDM": 56}
