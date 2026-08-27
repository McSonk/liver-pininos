"""Unit tests for the ``idssp.sonk.config`` module.

These tests cover config initialisation and validation, enum/flag serialisation,
the cgroup memory helpers, and the limited-environment detection logic. They rely
on mocked hardware detection, a no-op ``load_dotenv`` and controlled environment
variables so that no GPU, network, real dataset, or the repository ``.env`` file
is ever used.
"""

import builtins
import io
from dataclasses import replace

import pytest

from idssp.sonk import config as config_module
from idssp.sonk.config import AvailableModels, Config, Mode


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


def test_available_models_has_expected_values():
    """The model enum exposes the expected string values."""
    assert AvailableModels.U_NET.value == "u-net"
    assert AvailableModels.SEG_RES_NET.value == "seg-res-net"
    assert AvailableModels.SWIN_UNETR.value == "swin-unetr"


def test_mode_enum_has_expected_values():
    """The mode enum exposes the expected string values."""
    assert Mode.TRAIN.value == "train"
    assert Mode.TEST.value == "test"


# --------------------------------------------------------------------------- #
# Frozen dataclass behaviour
# --------------------------------------------------------------------------- #


def test_config_is_frozen(minimal_config):
    """Assigning to a field of a frozen Config raises FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        minimal_config.RUN_ID = "nope"


# --------------------------------------------------------------------------- #
# Singleton access
# --------------------------------------------------------------------------- #


def test_get_raises_when_not_initialised():
    """get() must raise RuntimeError before init() is called."""
    with pytest.raises(RuntimeError):
        config_module.get()


def test_get_returns_initialised_config(monkeypatch, minimal_config):
    """get() returns the module-level singleton once it is set."""
    monkeypatch.setattr(config_module, "_config", minimal_config)
    assert config_module.get() is minimal_config


# --------------------------------------------------------------------------- #
# is_limited_env()
# --------------------------------------------------------------------------- #


def test_is_limited_env_local_env_true(minimal_config):
    """Any local environment is limited."""
    assert config_module.is_limited_env(config=minimal_config) is True


def test_is_limited_env_cpu_device_true(minimal_config):
    """A CPU device is limited regardless of the reported environment."""
    cfg = replace(minimal_config, ENV="cloud", DEVICE="cpu", HC_GPU=True)
    assert config_module.is_limited_env(config=cfg) is True


def test_is_limited_env_low_vram_true(minimal_config):
    """A CUDA device with low VRAM is limited when include_vram is True."""
    cfg = replace(minimal_config, ENV="cloud", DEVICE="cuda", HC_GPU=False)
    assert config_module.is_limited_env(config=cfg) is True


def test_is_limited_env_not_limited_false(minimal_config):
    """A cloud CUDA high-VRAM environment is not limited."""
    cfg = replace(minimal_config, ENV="cloud", DEVICE="cuda", HC_GPU=True)
    assert config_module.is_limited_env(config=cfg) is False


def test_is_limited_env_ignore_vram_false(minimal_config):
    """Low VRAM is ignored when include_vram is False."""
    cfg = replace(minimal_config, ENV="cloud", DEVICE="cuda", HC_GPU=False)
    assert config_module.is_limited_env(config=cfg, include_vram=False) is False


# --------------------------------------------------------------------------- #
# to_dict()
# --------------------------------------------------------------------------- #


def test_to_dict_serializes_enum_to_string(minimal_config):
    """Enums are serialised to their string values."""
    d = config_module.to_dict(minimal_config)
    assert d["MODEL"] == "seg-res-net"
    assert d["MODE"] == "train"
    assert not isinstance(d["MODEL"], AvailableModels)


def test_to_dict_converts_tuples_to_lists(minimal_config):
    """Tuple fields are converted to lists."""
    d = config_module.to_dict(minimal_config)
    assert d["ISO_SPACING"] == [1.0, 1.0, 1.0]
    assert d["TRAIN_PATCH_SIZE"] == [64, 64, 64]
    assert d["VAL_PATCH_SIZE"] == [64, 64, 64]
    assert all(isinstance(v, list) for v in (d["ISO_SPACING"], d["TRAIN_PATCH_SIZE"], d["VAL_PATCH_SIZE"]))


def test_to_dict_converts_paths_to_strings(minimal_config):
    """Path fields are converted to strings."""
    d = config_module.to_dict(minimal_config)
    assert d["OUTPUT_DIR"] == str(minimal_config.OUTPUT_DIR)
    assert d["RUN_DIR"] == str(minimal_config.RUN_DIR)
    assert d["SPLIT_JSON"] == str(minimal_config.SPLIT_JSON)
    assert isinstance(d["OUTPUT_DIR"], str)


def test_to_dict_excludes_sensitive_fields(minimal_config):
    """Secrets and contact details are excluded from the serialised snapshot."""
    excluded = {
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "EMAIL_PASSWORD",
        "EMAIL_SENDER",
        "EMAIL_RECIPIENT",
        "SMTP_HOST",
        "SMTP_PORT",
    }
    d = config_module.to_dict(minimal_config)
    for key in excluded:
        assert key not in d


# --------------------------------------------------------------------------- #
# to_param_dict()
# --------------------------------------------------------------------------- #


def test_to_param_dict_uses_singleton(monkeypatch, minimal_config):
    """to_param_dict() serialises the singleton config."""
    monkeypatch.setattr(config_module, "_config", minimal_config)
    assert config_module.to_param_dict()  # does not raise


def test_to_param_dict_excludes_noisy_keys(monkeypatch, minimal_config):
    """Noisy/path keys are dropped from the parameter dict."""
    monkeypatch.setattr(config_module, "_config", minimal_config)
    excluded = {
        "RUN_ID",
        "ENV",
        "DEVICE",
        "MODE",
        "OUTPUT_DIR",
        "RUN_DIR",
        "SPLIT_JSON",
        "TRAIN_STATS_DIR",
        "PRE_TRAINED_MODEL_PATH",
        "LOG_LEVEL_CONSOLE",
        "LOG_LEVEL_FILE",
    }
    d = config_module.to_param_dict()
    for key in excluded:
        assert key not in d


def test_to_param_dict_converts_lists_to_strings(monkeypatch, minimal_config):
    """List/tuple values are converted to strings in the parameter dict."""
    monkeypatch.setattr(config_module, "_config", minimal_config)
    d = config_module.to_param_dict()
    assert d["DICE_CE_WEIGHTS"] == "[0.5, 1.0, 3.0]"
    assert d["ISO_SPACING"] == "[1.0, 1.0, 1.0]"


def test_to_param_dict_removes_none_values(monkeypatch, minimal_config):
    """None values are dropped from the parameter dict."""
    monkeypatch.setattr(config_module, "_config", minimal_config)
    d = config_module.to_param_dict()
    assert None not in d.values()


# --------------------------------------------------------------------------- #
# get_cgroup_memory_limit_bytes()
# --------------------------------------------------------------------------- #


def _patch_cgroup_files(monkeypatch, files):
    """Patch open and os.path.exists to mimic the given cgroup file map."""

    def fake_open(path, *args, **kwargs):
        if path in files:
            return io.StringIO(files[path])
        raise FileNotFoundError(path)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(config_module.os.path, "exists", lambda p: p in files)


def test_cgroup_limit_v2_numeric():
    """A numeric cgroup v2 limit is returned directly."""
    import os as _os

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config_module.os.path, "exists", lambda p: p in {"/sys/fs/cgroup/memory.max": "1"})
    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: io.StringIO("8589934592") if p.endswith("memory.max") else (_ for _ in ()).throw(FileNotFoundError(p)))
    assert config_module.get_cgroup_memory_limit_bytes() == 8589934592
    monkeypatch.undo()


def test_cgroup_limit_v2_max_falls_back_to_v1(monkeypatch):
    """A cgroup v2 'max' value falls through to the v1 file."""
    files = {
        "/sys/fs/cgroup/memory.max": "max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "4096",
    }
    _patch_cgroup_files(monkeypatch, files)
    assert config_module.get_cgroup_memory_limit_bytes() == 4096


def test_cgroup_limit_v1_present(monkeypatch):
    """A present cgroup v1 file is returned when v2 is missing."""
    files = {
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "2048",
    }
    _patch_cgroup_files(monkeypatch, files)
    assert config_module.get_cgroup_memory_limit_bytes() == 2048


def test_cgroup_limit_missing_returns_minus_one(monkeypatch):
    """Missing cgroup files yield -1."""
    _patch_cgroup_files(monkeypatch, {})
    assert config_module.get_cgroup_memory_limit_bytes() == -1


# --------------------------------------------------------------------------- #
# get_container_usage()
# --------------------------------------------------------------------------- #


def test_container_usage_v2_limited(monkeypatch):
    """cgroup v2 limited files produce a full usage tuple."""
    files = {
        "/sys/fs/cgroup/memory.max": "8589934592",
        "/sys/fs/cgroup/memory.current": "4294967296",
    }
    _patch_cgroup_files(monkeypatch, files)
    limit_gb, usage_gb, free_gb, usage_pct = config_module.get_container_usage()
    assert limit_gb == 8.0
    assert usage_gb == 4.0
    assert free_gb == 4.0
    assert usage_pct == 50.0


def test_container_usage_v2_unlimited(monkeypatch):
    """cgroup v2 'max' returns the unknown sentinel tuple."""
    files = {
        "/sys/fs/cgroup/memory.max": "max",
        "/sys/fs/cgroup/memory.current": "1",
    }
    _patch_cgroup_files(monkeypatch, files)
    assert config_module.get_container_usage() == (-1, -1, -1, -1)


def test_container_usage_v1_unlimited_sentinel(monkeypatch):
    """cgroup v1 unlimited sentinel returns the unknown sentinel tuple."""
    files = {
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1",
    }
    _patch_cgroup_files(monkeypatch, files)
    assert config_module.get_container_usage() == (-1, -1, -1, -1)


def test_container_usage_missing_files(monkeypatch):
    """Missing cgroup files return the unknown sentinel tuple."""
    _patch_cgroup_files(monkeypatch, {})
    assert config_module.get_container_usage() == (-1, -1, -1, -1)


def test_container_usage_invalid_value(monkeypatch):
    """A non-numeric cgroup value returns the unknown sentinel tuple."""
    files = {
        "/sys/fs/cgroup/memory.max": "not-a-number",
        "/sys/fs/cgroup/memory.current": "1",
    }
    _patch_cgroup_files(monkeypatch, files)
    assert config_module.get_container_usage() == (-1, -1, -1, -1)


# --------------------------------------------------------------------------- #
# init() helpers
# --------------------------------------------------------------------------- #


def _setup_valid_env(monkeypatch, tmp_path, **overrides):
    """Set a complete, valid set of environment variables under tmp_path."""
    output_dir = tmp_path / "output"
    stats_dir = tmp_path / "stats"
    ct_root = tmp_path / "ct_root"
    ct_test = tmp_path / "ct_test"
    split_json = tmp_path / "splits" / "split.json"
    for path in (output_dir, stats_dir, ct_root, ct_test):
        path.mkdir(parents=True, exist_ok=True)
    split_json.parent.mkdir(parents=True, exist_ok=True)

    env_vars = {
        "PIN_ENV": "local",
        "LITS_CT_ROOT": str(ct_root),
        "LITS_CT_TEST": str(ct_test),
        "OUTPUT_DIR": str(output_dir),
        "STATS_DIR": str(stats_dir),
        "SPLIT_JSON": str(split_json),
    }
    env_vars.update(overrides)
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


def _apply_hardware_patches(monkeypatch, total_bytes, cgroup_limit=-1):
    """Patch load_dotenv, CUDA detection, and memory detection."""

    class _VirtualMemory:
        def __init__(self, total):
            self.total = total

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(config_module.psutil, "virtual_memory", lambda: _VirtualMemory(total_bytes))
    monkeypatch.setattr(config_module, "get_cgroup_memory_limit_bytes", lambda: cgroup_limit)


def _init_valid(monkeypatch, tmp_path, total_gb=200.0, **env_overrides):
    _setup_valid_env(monkeypatch, tmp_path, **env_overrides)
    _apply_hardware_patches(monkeypatch, int(total_gb * (1024 ** 3)))
    return config_module.init()


# --------------------------------------------------------------------------- #
# init(): validation
# --------------------------------------------------------------------------- #


def test_init_missing_pin_env_raises_environment_error(monkeypatch, tmp_path):
    """init() raises EnvironmentError when PIN_ENV is not set."""
    _setup_valid_env(monkeypatch, tmp_path)
    _apply_hardware_patches(monkeypatch, 200 * (1024 ** 3))
    monkeypatch.delenv("PIN_ENV", raising=False)
    with pytest.raises(EnvironmentError):
        config_module.init()


def test_init_invalid_pin_env_raises_value_error(monkeypatch, tmp_path):
    """init() raises ValueError for an unrecognised PIN_ENV."""
    _setup_valid_env(monkeypatch, tmp_path, PIN_ENV="bogus")
    _apply_hardware_patches(monkeypatch, 200 * (1024 ** 3))
    with pytest.raises(ValueError):
        config_module.init()


@pytest.mark.parametrize("missing_var", [
    "LITS_CT_ROOT",
    "LITS_CT_TEST",
    "OUTPUT_DIR",
    "STATS_DIR",
    "SPLIT_JSON",
])
def test_init_missing_required_variable_raises_value_error(monkeypatch, tmp_path, missing_var):
    """Each required environment variable is validated by init()."""
    _setup_valid_env(monkeypatch, tmp_path)
    _apply_hardware_patches(monkeypatch, 200 * (1024 ** 3))
    monkeypatch.delenv(missing_var, raising=False)
    with pytest.raises(ValueError):
        config_module.init()


def test_init_missing_ct_root_dir_raises_file_not_found(monkeypatch, tmp_path):
    """init() raises FileNotFoundError when the CT root does not exist."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    env_vars = {
        "PIN_ENV": "local",
        "LITS_CT_ROOT": str(tmp_path / "does_not_exist"),
        "LITS_CT_TEST": str(tmp_path / "ct_test"),
        "OUTPUT_DIR": str(output_dir),
        "STATS_DIR": str(tmp_path / "stats"),
        "SPLIT_JSON": str(tmp_path / "split.json"),
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    _apply_hardware_patches(monkeypatch, 200 * (1024 ** 3))
    with pytest.raises(FileNotFoundError):
        config_module.init()


# --------------------------------------------------------------------------- #
# init(): fallbacks
# --------------------------------------------------------------------------- #


def test_init_invalid_log_levels_fall_back(monkeypatch, tmp_path):
    """Invalid log levels fall back to INFO and DEBUG respectively."""
    cfg = _init_valid(
        monkeypatch,
        tmp_path,
        LOG_LEVEL_CONSOLE="bogus",
        LOG_LEVEL_FILE="bogus",
    )
    assert cfg.LOG_LEVEL_CONSOLE == "INFO"
    assert cfg.LOG_LEVEL_FILE == "DEBUG"


def test_init_invalid_cache_source_falls_back_to_disk_low_ram(monkeypatch, tmp_path):
    """An invalid cache source defaults to 'ram', then disk when RAM is low."""
    cfg = _init_valid(
        monkeypatch,
        tmp_path,
        total_gb=50.0,
        CACHE_TRAIN_SOURCE="bogus",
        CACHE_VAL_SOURCE="bogus",
    )
    assert cfg.USE_CACHE_TRAIN_DATASET is False
    assert cfg.USE_CACHE_VAL_DATASET is False


def test_init_cache_ram_stays_ram_high_ram(monkeypatch, tmp_path):
    """A ram cache source stays ram when RAM is high."""
    cfg = _init_valid(monkeypatch, tmp_path, total_gb=200.0)
    assert cfg.USE_CACHE_TRAIN_DATASET is True
    assert cfg.USE_CACHE_VAL_DATASET is True


def test_init_cache_ram_falls_back_to_disk_low_ram(monkeypatch, tmp_path):
    """A ram cache source falls back to disk when RAM is low."""
    cfg = _init_valid(monkeypatch, tmp_path, total_gb=50.0)
    assert cfg.USE_CACHE_TRAIN_DATASET is False
    assert cfg.USE_CACHE_VAL_DATASET is False


def test_init_creates_run_and_log_directories(monkeypatch, tmp_path):
    """init() creates the run and log directories under OUTPUT_DIR."""
    cfg = _init_valid(monkeypatch, tmp_path, total_gb=200.0)
    assert cfg.RUN_DIR.exists()
    assert cfg.LOG_DIR.exists()


# --------------------------------------------------------------------------- #
# init(): notifications
# --------------------------------------------------------------------------- #


def test_init_email_enabled_missing_fields_raises(monkeypatch, tmp_path):
    """Email enabled with missing fields raises ValueError."""
    _setup_valid_env(monkeypatch, tmp_path, ENABLE_EMAIL_NOTIFICATIONS="true")
    _apply_hardware_patches(monkeypatch, 200 * (1024 ** 3))
    with pytest.raises(ValueError):
        config_module.init()


def test_init_email_smtp_port_not_integer_raises(monkeypatch, tmp_path):
    """A non-integer SMTP_PORT raises ValueError."""
    _setup_valid_env(
        monkeypatch,
        tmp_path,
        ENABLE_EMAIL_NOTIFICATIONS="true",
        SMTP_HOST="smtp.example.com",
        SMTP_PORT="abc",
        EMAIL_SENDER="sender",
        EMAIL_PASSWORD="pw",
        EMAIL_RECIPIENT="recipient",
    )
    _apply_hardware_patches(monkeypatch, 200 * (1024 ** 3))
    with pytest.raises(ValueError):
        config_module.init()


def test_init_telegram_enabled_missing_fields_raises(monkeypatch, tmp_path):
    """Telegram enabled with missing fields raises ValueError."""
    _setup_valid_env(monkeypatch, tmp_path, ENABLE_TELEGRAM_NOTIFICATIONS="true")
    _apply_hardware_patches(monkeypatch, 200 * (1024 ** 3))
    with pytest.raises(ValueError):
        config_module.init()


# --------------------------------------------------------------------------- #
# init(): re-initialisation
# --------------------------------------------------------------------------- #


def test_init_reinit_same_mode_returns_same_instance(monkeypatch, tmp_path):
    """Re-initialising with the same mode returns the same instance."""
    cfg1 = _init_valid(monkeypatch, tmp_path, total_gb=200.0)
    cfg2 = config_module.init(mode=Mode.TRAIN)
    assert cfg1 is cfg2


def test_init_reinit_different_mode_raises(monkeypatch, tmp_path):
    """Re-initialising with a different mode raises RuntimeError."""
    _init_valid(monkeypatch, tmp_path, total_gb=200.0)
    with pytest.raises(RuntimeError):
        config_module.init(mode=Mode.TEST)
