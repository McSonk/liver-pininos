"""Configuration file for the HCC Segmentation Thesis.
Adjust paths and hyperparameters as needed.
"""
import datetime
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import torch
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
     # Environment & Device
    RUN_ID: str
    ENV: str
    DEVICE: str
    HC_GPU: bool
    '''HC_GPU is a flag to indicate if we are on the High-Compute GPU.
    Note that this only means the GPU has more than 30GB of VRAM'''
    RANDOM_SEED: int = 42
    cpu_memory: float = -1.0
    container_memory: float = -1.0
    '''The total memory available to the process, in GB. This takes into account
       cgroup limits, so if the process is running in a container with limited memory,
       this will reflect that limit rather than the total RAM of the host machine.'''

    # Preprocessing
    HU_WINDOW_MIN: int = -175
    HU_WINDOW_MAX: int = 250
    ISO_SPACING: tuple = (1.0, 1.0, 1.0)
    '''The isotropic spacing to which all CT volumes will be resampled.
       A good choice is (1.5, 1.5, 1.5). It is memory efficient but it might introduce
       some blurring. (1.0, 1.0, 1.0) is an optimal choice.
       Please note that in LiTS most of the volumes have a z spacing of around 0.7-1.0mm, 
       so resampling to 1.0mm will not introduce much blurring while ensuring. However,
       there are 2 volumes with a z spacing of .5mm, so here we should be more careful.'''

    TRAIN_PATCH_SIZE: tuple = (96, 96, 96)
    VAL_PATCH_SIZE: tuple = (96, 96, 96)
    '''The size of the 3D patches to be extracted from the volumes for training and validation.
       (Only used in local env). NOT TO BE CONFUSED WITH `VAL_BATCH_SIZE`'''
    USE_CACHE_TRAIN_DATASET: bool = True
    '''Whether to use a caching dataset that keeps preprocessed volumes in memory.
    This can speed up training but requires more RAM.
    If False, PersistentDataset will be used instead
    '''
    USE_CACHE_VAL_DATASET: bool = True
    '''Whether to use a caching dataset that keeps preprocessed volumes in memory.
    This can speed up training but requires more RAM.
    If False, PersistentDataset will be used instead
    '''

    # Training
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 2
    '''DataLoader's batch size. Set to 1 for memory safety, especially with large 3D volumes.
       (Usually between 1 and 4 depending on GPU VRAM)
       Note that the final batch size will be `BATCH_SIZE` * `RAND_CROP_NUM_SAMPLES`'''
    VAL_BATCH_SIZE: int = 1
    '''DataLoader's batch size for validation. Kept at 1 for deterministic evaluation
    and memory safety with large 3D volumes. NOT TO BE CONFUSED WITH `VAL_PATCH_SIZE`'''
    RAND_CROP_NUM_SAMPLES: int = 2
    '''Number of random crops to extract from each volume during training.
    Note that the final batch size will be `BATCH_SIZE` * `RAND_CROP_NUM_SAMPLES`
    '''
    CACHE_NUM_WORKERS: int = -1
    '''Number of parallel processes for data loading (Only useful if using CacheDataset).'''
    DL_NUM_WORKERS: int = -1
    '''Number of parallel processes for data loading in the DataLoader. Set to 0
    for debugging or if you encounter issues with multiprocessing. A good default
    is the number of CPU cores minus one.'''
    PIN_MEMORY: bool = True
    NUM_EPOCHS: int = 150
    NUM_CLASSES: int = -1
    '''How many classes to predict.
    For binary segmentation, set to 2 (tumour vs background).
    For multi-class, set to 3 (background, liver, tumour).'''
    TUMOUR_CLASS_INDEX: int = -1
    '''The index of the tumour class in the model's output channels.
    For binary segmentation (NUM_CLASSES=2), this should be 1 if the
    classes are ordered as [background, tumour]. For multi-class segmentation
    (NUM_CLASSES=3), this should be 2 if the classes are ordered as [background,
    liver, tumour].'''
    DICE_CE_WEIGHTS: list = None
    '''The weights for the combined Dice + Cross-Entropy loss.
    This should be a list of length `NUM_CLASSES`, where the value at `TUMOUR_CLASS_INDEX`
    is higher to emphasise learning the tumour class. For example, for `NUM_CLASSES=3`
    and `TUMOUR_CLASS_INDEX=2`,  a good choice is `[0.0, 1.0, 3.0]` to ignore the
    background, give some weight to the liver, and more weight to the tumour.
    '''

    # Early Stopping
    EARLY_STOPPING_PATIENCE: int = 30
    '''Number of epochs with no improvement after which training will be stopped.'''
    EARLY_STOPPING_MIN_DELTA: float = 0.005
    '''Minimum change in the monitored metric to qualify as an improvement.'''
    WARMUP_EPOCHS: int = 5
    '''Number of epochs for linear learning rate warmup (CosineSchedule).'''

    # Paths (resolved at init)
    CT_ROOT: Path = field(default_factory=Path)
    CT_TEST: Path = field(default_factory=Path)
    OUTPUT_DIR: Path = field(default_factory=Path)
    CHECKPOINT_DIR: Path = field(default_factory=Path)
    LOG_DIR: Path = field(default_factory=Path)
    TENSORBOARD_DIR: Path = field(default_factory=Path)
    PERSISTENT_DATASET_DIR: Path = field(default_factory=Path)
    STATS_DIR: Path = field(default_factory=Path)
    SPLIT_DIR: Path = field(default_factory=Path)
    TRAIN_STATS_DIR: Path = field(default_factory=Path)
    PER_CASE_TRAIN_STATS_FILE: Path = field(default_factory=Path)
    LOG_LEVEL_CONSOLE: str = "INFO"
    LOG_LEVEL_FILE: str = "DEBUG"

    # Notifications (mail)
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    SMTP_HOST: str = ""
    SMTP_PORT: int = -1
    EMAIL_SENDER: str = ""
    EMAIL_PASSWORD: str = ""
    EMAIL_RECIPIENT: str = ""

    # Notifications (Telegram)
    ENABLE_TELEGRAM_NOTIFICATIONS: bool = False
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

# Module-level singleton (lazy)
_config: Config = None

def init() -> Config:
    global _config
    if _config is not None:
        return _config

    #  Check computing power
    # ----------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hc_gpu = False
    cpu_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
    container_memory_bytes = get_cgroup_memory_limit_bytes()
    container_memory = (
        container_memory_bytes / (1024 ** 3) if container_memory_bytes > 0 else -1.0
    )  # GB; -1.0 means unknown/unlimited
    process_memory_limit = container_memory if container_memory > 0 else cpu_memory
    lots_of_ram = process_memory_limit >= 50

    if device == "cuda":
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hc_gpu = vram_gb >= 30

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    cpu_count = os.cpu_count() or 1

    # ---------------------------------------
    # YOU CAN CHANGE VALUES HERE
    # ---------------------------------------
    # `hc_gpu` is a flag to indicate if we are on the High-Compute GPU.
    # Note that this only means the GPU has more than 30GB of VRAM.

    # num_classes = 2 or 3
    num_classes = 3
    tumour_class_index = 2 if num_classes == 3 else 1
    dice_ce_weights = [0.0, 1.0, 3.0] if num_classes == 3 else [1.0, 3.0]
    gpu_num_workers = 8 if hc_gpu else 2

    local_specific = {
        "cache_num_workers": 0,
        "dl_num_workers": 0,
        "pin_memory": False,
        "batch_size": 1,
        "num_epochs": 5,
        "train_patch_size": (64, 64, 64),
        "val_patch_size": (64, 64, 64),
        "iso_spacing": (2.0, 2.0, 2.0),
    }

    cloud_specific = {
        "cache_num_workers": 4 if process_memory_limit > 60 else 2,
        "dl_num_workers": min(gpu_num_workers, cpu_count),
        "pin_memory": True,
        "batch_size": 4 if hc_gpu else 2,
        "num_epochs": 200 if hc_gpu else 5,
        "train_patch_size": (96, 96, 96),
        "val_patch_size": (96, 96, 96),  # Not used but kept for config/logging consistency
        # TODO: Tune. Both options sound valid, so decide which is better based on experiments.
        "iso_spacing": (1.0, 1.0, 1.0) if hc_gpu else (1.5, 1.5, 1.5),
    }

    # -----------------------------------------------------------------------------
    # 0. Load Environment Variables
    # -----------------------------------------------------------------------------

    print("=" * 80)
    print("This is a configuration file.")
    print("The configuration is loaded at the start of the program and defines important constants, ")
    print("paths, and hyperparameters.")
    print("Please review the settings below and adjust them as needed before running the program.")
    print("If you are running this for the first time, make sure to create a .env")
    print("file based on the .env.example template and fill in the required paths and settings.")
    print("")
    print("NOTE: The logs of this file won't be stored in the logs directory,")
    print("so please pay attention to any warnings or errors printed here.")
    print("=" * 80)
    print("[Config] Loading environment variables from .env file...")
    load_dotenv()

    # -----------------------------------------------------------------------------
    # 1. Suppress warnings from libraries to keep the logs clean.
    # -----------------------------------------------------------------------------
    warnings.filterwarnings(
        "ignore",
        # MONAI sliding-window inference triggers a deprecation warning on PyTorch's internal
        # index behaviour. This can be safely ignored.
        message=".*non-tuple sequence for multidimensional indexing.*",
        category=DeprecationWarning,
        module=r"torch(\..*)?$",
    )

    # -----------------------------------------------------------------------------
    # 2. Environment detection and validation
    # -----------------------------------------------------------------------------
    env = os.getenv("PIN_ENV", "").lower()

    if not env:
        raise EnvironmentError(
            "[Config] ERROR: Environment variable 'PIN_ENV' is not set!\n"
            "Please do one of the following:\n"
            "   1. Create a '.env' file in the project root with: PIN_ENV=local\n"
            "   2. Or set it in your terminal: export PIN_ENV=local\n"
            "   3. Or set it in Lightning AI Studio settings.\n\n"
            "In any case, be sure to check .env.example for the expected format "
            "of the .env file and required variables."
        )

    recognised_envs = {"local", "cloud"}

    if env not in recognised_envs:
        raise ValueError(
            f"[Config] Environment [{env}] is not recognised. Please set PIN_ENV to"
            f" one of {recognised_envs}"
        )

    print(f"[Config] Loading configuration for environment: [{env.upper()}]")

    # -----------------------------------------------------------------------------
    # 2. Shared Constants (Same across all environments)
    # -----------------------------------------------------------------------------

    #  File locations
    # ----------------------------------------------------------------

    ct_root_str = os.getenv("LITS_CT_ROOT")
    ct_test_str = os.getenv("LITS_CT_TEST")
    output_dir_str = os.getenv("OUTPUT_DIR")
    # CHECKPOINT_DIR_STR = os.getenv("CHECKPOINT_DIR")
    persistent_dataset_dir_str = os.getenv("PERSISTENT_DATASET_DIR")
    stats_dir_str = os.getenv("STATS_DIR")
    split_dir_str = os.getenv("SPLIT_DIR")

    # LOG_DIR_STR = os.getenv("LOG_DIR")
    log_level_console = os.getenv("LOG_LEVEL_CONSOLE", "INFO").upper()
    log_level_file = os.getenv("LOG_LEVEL_FILE", "DEBUG").upper()

    #  File validations
    # ----------------------------------------------------------------

    if not ct_root_str:
        raise ValueError("[Config] Environment variable 'LITS_CT_ROOT' is not set!")
    if not ct_test_str:
        raise ValueError("[Config] Environment variable 'LITS_CT_TEST' is not set!")
    if not output_dir_str:
        raise ValueError("[Config] Environment variable 'OUTPUT_DIR' is not set. "
            "Please set 'OUTPUT_DIR' to the directory where checkpoints, logs, and "
            "other outputs will be saved.")
    if not persistent_dataset_dir_str:
        print("[Config] Warning: 'PERSISTENT_DATASET_DIR' is not set. "
            "PersistentDataset will be disabled.")

    if not stats_dir_str:
        raise ValueError("[Config] Environment variable 'STATS_DIR' is not set. "
            "Stratification cannot be performed. Please set 'STATS_DIR' to the "
            "directory where the precomputed dataset statistics are stored.")

    if not split_dir_str:
        raise ValueError("[Config] Environment variable 'SPLIT_DIR' is not set. "
            "Splitting cannot be performed. Please set 'SPLIT_DIR' to the "
            "directory where the split files will be stored.")

    if log_level_console not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        print(f"[Config] Warning: LOG_LEVEL_CONSOLE '{log_level_console}' is not "
            "valid. Defaulting to 'INFO'.")
        log_level_console = "INFO"
    if log_level_file not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        print(f"[Config] Warning: LOG_LEVEL_FILE '{log_level_file}' is not valid. "
            "Defaulting to 'DEBUG'.")
        log_level_file = "DEBUG"

    #  Hyperparameters and constants
    # ----------------------------------------------------------------
    if num_classes != 2 and num_classes != 3:
        raise ValueError("[Config] NUM_CLASSES must be either 2 (for binary segmentation) "
                        "or 3 (for multi-class segmentation).")

    cache_train_source = os.getenv("CACHE_TRAIN_SOURCE", "ram").lower()
    cache_val_source = os.getenv("CACHE_VAL_SOURCE", "ram").lower()

    if cache_train_source not in {"ram", "disk"}:
        print(f"[Config] Warning: CACHE_TRAIN_SOURCE '{cache_train_source}' is not valid. "
            "Defaulting to 'ram'.")
        cache_train_source = "ram"

    if cache_val_source not in {"ram", "disk"}:
        print(f"[Config] Warning: CACHE_VAL_SOURCE '{cache_val_source}' is not valid. "
            "Defaulting to 'ram'.")
        cache_val_source = "ram"

    if cache_train_source == "ram" and not lots_of_ram:
        print("[Config] Warning: CACHE_TRAIN_SOURCE is set to 'ram' but not enough CPU "
              "RAM is available. Defaulting to 'disk'.")
        cache_train_source = "disk"

    if cache_val_source == "ram" and not lots_of_ram:
        print("[Config] Warning: CACHE_VAL_SOURCE is set to 'ram' but not enough CPU "
              "RAM is available. Defaulting to 'disk'.")
        cache_val_source = "disk"

    use_cache_train_dataset = cache_train_source == "ram"
    use_cache_val_dataset = cache_val_source == "ram"

    # -------------------
    # Path resolution
    # -------------------

    ct_root = Path(ct_root_str)
    ct_test = Path(ct_test_str)
    output_dir = Path(output_dir_str)
    stats_dir = Path(stats_dir_str)
    split_dir = Path(split_dir_str)
    train_stats_dir = stats_dir / "train"
    train_stats_dir.mkdir(parents=True, exist_ok=True)
    per_case_train_stats_file = train_stats_dir / "per_case_summary.csv"
    persistent_dataset_dir = Path(persistent_dataset_dir_str) if persistent_dataset_dir_str else None

    checkpoint_dir = output_dir / run_id / "checkpoints"
    log_dir = output_dir / run_id / "logs"
    tensorboard_dir = output_dir / run_id / "tensorboard"

    if device == "cuda" and (not use_cache_train_dataset or not use_cache_val_dataset):
        if persistent_dataset_dir is None:
            raise ValueError("[Config] Persistent dataset directory must be set when"
                            " using CUDA without cache dataset.")

    # -----------------------------------------------------------------------------
    # 3. Environment-Specific Configuration
    # -----------------------------------------------------------------------------
    # Run `nproc` or `lscpu` for number of CPU cores. NUM_WORKERS < number of cores.

    if env == "local":
        print("[Config] Running in LOCAL environment.")
        cache_num_workers = local_specific["cache_num_workers"]
        dl_num_workers = local_specific["dl_num_workers"]
        pin_memory = local_specific["pin_memory"]
        batch_size = local_specific["batch_size"]
        num_epochs = local_specific["num_epochs"]
        train_patch_size = local_specific["train_patch_size"]
        val_patch_size = local_specific["val_patch_size"]
        iso_spacing = local_specific["iso_spacing"]

    else:
        print("[Config] Running in CLOUD environment. Using more computing power.")
        cache_num_workers = cloud_specific["cache_num_workers"]
        dl_num_workers = cloud_specific["dl_num_workers"]
        pin_memory = cloud_specific["pin_memory"]
        batch_size = cloud_specific["batch_size"]
        num_epochs = cloud_specific["num_epochs"]
        train_patch_size = cloud_specific["train_patch_size"]
        val_patch_size = cloud_specific["val_patch_size"]
        iso_spacing = cloud_specific["iso_spacing"]

    # -----------------------------------------------------------------------------
    # 4. Final Safety Check & Directory Creation
    # -----------------------------------------------------------------------------
    log_dir.mkdir(parents=True, exist_ok=True)

    if not ct_root.exists():
        raise FileNotFoundError(f"[Config] CT root directory does not exist: {ct_root}")

    if persistent_dataset_dir and not persistent_dataset_dir.exists():
        print("[Config] Persistent dataset directory does not exist. "
            f"Creating: {persistent_dataset_dir}")
        persistent_dataset_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------------
    # 5. Email / Notification Configuration
    # -----------------------------------------------------------------------------
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port_raw = os.getenv("SMTP_PORT", "").strip()
    smtp_port = 0
    email_sender = os.getenv("EMAIL_SENDER", "")
    email_password = os.getenv("EMAIL_PASSWORD", "")
    email_recipient = os.getenv("EMAIL_RECIPIENT", "")
    enable_email_notifications = os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "false").lower() == "true"

    enable_telegram_notifications = os.getenv(
        "ENABLE_TELEGRAM_NOTIFICATIONS", "false").lower() == "true"
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if enable_email_notifications:
        if not all([smtp_host, smtp_port_raw, email_sender, email_password, email_recipient]):
            raise ValueError(
                "[Config] Email notifications are enabled but one or more email configuration "
                "variables are missing. Please set SMTP_HOST, SMTP_PORT, EMAIL_SENDER, "
                "EMAIL_PASSWORD, and EMAIL_RECIPIENT in your .env file."
            )
        try:
            smtp_port = int(smtp_port_raw)
        except ValueError as exc:
            raise ValueError(
                "[Config] Email notifications are enabled but SMTP_PORT is not a valid integer. "
                "Please set SMTP_PORT to a numeric port value in your .env file."
            ) from exc
        print("[Config] Email notifications are enabled. Emails will be sent to "
            f"{email_recipient} at the end of training and for exceptions handled during training.")
    else:
        print("[Config] Email notifications are disabled. To enable, set "
                "ENABLE_EMAIL_NOTIFICATIONS=true and provide the required email "
                "configuration in the .env file.")

    if enable_telegram_notifications:
        if not all([telegram_bot_token, telegram_chat_id]):
            raise ValueError(
                "[Config] Telegram notifications are enabled but TELEGRAM_BOT_TOKEN and/or "
                "TELEGRAM_CHAT_ID is missing. Please set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in your .env file."
            )
        print("[Config] Telegram notifications are enabled. Alerts will be sent to "
            "the specified chat at the start of training, at the end of training, "
            "and for exceptions handled during training.")

    print("[Config] Configuration successfully loaded.")
    print("=" * 80)

    _config = Config(
        cpu_memory=cpu_memory,
        container_memory=container_memory,
        RUN_ID=run_id,
        ENV=env,
        DEVICE=device,
        HC_GPU=hc_gpu,
        CT_ROOT=ct_root,
        CT_TEST=ct_test,
        NUM_CLASSES=num_classes,
        DICE_CE_WEIGHTS=dice_ce_weights,
        OUTPUT_DIR=output_dir,
        CHECKPOINT_DIR=checkpoint_dir,
        LOG_DIR=log_dir,
        TENSORBOARD_DIR=tensorboard_dir,
        PERSISTENT_DATASET_DIR=persistent_dataset_dir,
        STATS_DIR=stats_dir,
        SPLIT_DIR=split_dir,
        TRAIN_STATS_DIR=train_stats_dir,
        PER_CASE_TRAIN_STATS_FILE=per_case_train_stats_file,
        LOG_LEVEL_CONSOLE=log_level_console,
        LOG_LEVEL_FILE=log_level_file,
        CACHE_NUM_WORKERS=cache_num_workers,
        DL_NUM_WORKERS=dl_num_workers,
        PIN_MEMORY=pin_memory,
        BATCH_SIZE=batch_size,
        NUM_EPOCHS=num_epochs,
        TRAIN_PATCH_SIZE=train_patch_size,
        VAL_PATCH_SIZE=val_patch_size,
        ISO_SPACING=iso_spacing,
        USE_CACHE_TRAIN_DATASET=use_cache_train_dataset,
        USE_CACHE_VAL_DATASET=use_cache_val_dataset,
        TUMOUR_CLASS_INDEX=tumour_class_index,
        ENABLE_EMAIL_NOTIFICATIONS=enable_email_notifications,
        SMTP_HOST=smtp_host,
        SMTP_PORT=smtp_port,
        EMAIL_SENDER=email_sender,
        EMAIL_PASSWORD=email_password,
        EMAIL_RECIPIENT=email_recipient,
        ENABLE_TELEGRAM_NOTIFICATIONS=enable_telegram_notifications,
        TELEGRAM_BOT_TOKEN=telegram_bot_token,
        TELEGRAM_CHAT_ID=telegram_chat_id,
    )

    return _config

def get() -> Config:
    """Safely retrieve the initialised config. Raises if init() was never called."""
    if _config is None:
        raise RuntimeError(
            "Configuration not initialised. Call config.init() before accessing attributes."
        )
    return _config

# -----------------------------------------------------------------------------
# 6. Helper Functions
# -----------------------------------------------------------------------------

def is_limited_env(include_vram=True) -> bool:
    '''
    Returns True if the current environment is a limited resource
    environment (e.g., local with no GPU).

    if `include_vram=True` (default) it also takes into consideration
    the amount of memory of GPU so a CUDA device with less than 30GB of VRAM
    will be considered a limited environment.
    '''
    config = get()
    if config.ENV == "local" or config.DEVICE == "cpu":
        return True

    return include_vram and config.HC_GPU is False

def to_dict() -> dict:
    """Returns a serialisable snapshot of all configuration constants."""
    config = get()
    return {
        "RUN_ID": config.RUN_ID,
        "cpu_memory": config.cpu_memory,
        "container_memory": config.container_memory,
        # Environment & Device
        "ENV": config.ENV,
        "DEVICE": config.DEVICE,
        "HC_GPU": config.HC_GPU,
        "RANDOM_SEED": config.RANDOM_SEED,

        # Preprocessing
        "HU_WINDOW_MIN": config.HU_WINDOW_MIN,
        "HU_WINDOW_MAX": config.HU_WINDOW_MAX,
        "ISO_SPACING": list(config.ISO_SPACING),  # tuple → list for JSON/weights compatibility
        "TRAIN_PATCH_SIZE": list(config.TRAIN_PATCH_SIZE),
        "VAL_PATCH_SIZE": list(config.VAL_PATCH_SIZE),
        "USE_CACHE_TRAIN_DATASET": config.USE_CACHE_TRAIN_DATASET,
        "USE_CACHE_VAL_DATASET": config.USE_CACHE_VAL_DATASET,

        # Training Hyperparameters
        "LEARNING_RATE": config.LEARNING_RATE,
        "BATCH_SIZE": config.BATCH_SIZE,
        "VAL_BATCH_SIZE": config.VAL_BATCH_SIZE,
        "RAND_CROP_NUM_SAMPLES": config.RAND_CROP_NUM_SAMPLES,
        "CACHE_NUM_WORKERS": config.CACHE_NUM_WORKERS,
        "DL_NUM_WORKERS": config.DL_NUM_WORKERS,
        "PIN_MEMORY": config.PIN_MEMORY,
        "NUM_EPOCHS": config.NUM_EPOCHS,
        "NUM_CLASSES": config.NUM_CLASSES,
        "DICE_CE_WEIGHTS": config.DICE_CE_WEIGHTS,
        "TUMOUR_CLASS_INDEX": config.TUMOUR_CLASS_INDEX,

        # Early Stopping
        "EARLY_STOPPING_PATIENCE": config.EARLY_STOPPING_PATIENCE,
        "EARLY_STOPPING_MIN_DELTA": config.EARLY_STOPPING_MIN_DELTA,
        "WARMUP_EPOCHS": config.WARMUP_EPOCHS,

        # Paths (convert Path objects to strings)
        "CT_ROOT": str(config.CT_ROOT),
        "CT_TEST": str(config.CT_TEST),
        "OUTPUT_DIR": str(config.OUTPUT_DIR),
        "CHECKPOINT_DIR": str(config.CHECKPOINT_DIR),
        "TENSORBOARD_DIR": str(config.TENSORBOARD_DIR),
        "PERSISTENT_DATASET_DIR": str(config.PERSISTENT_DATASET_DIR) if config.PERSISTENT_DATASET_DIR else None,
        "STATS_DIR": str(config.STATS_DIR) if config.STATS_DIR else None,
        "LOG_DIR": str(config.LOG_DIR),
        "LOG_LEVEL_CONSOLE": config.LOG_LEVEL_CONSOLE,
        "LOG_LEVEL_FILE": config.LOG_LEVEL_FILE,
        "SPLIT_DIR": str(config.SPLIT_DIR),
        "TRAIN_STATS_DIR": str(config.TRAIN_STATS_DIR),
        "PER_CASE_TRAIN_STATS_FILE": str(config.PER_CASE_TRAIN_STATS_FILE),

        # Notifications (exclude contact details/secrets from persisted config snapshots)
        "ENABLE_EMAIL_NOTIFICATIONS": config.ENABLE_EMAIL_NOTIFICATIONS,
        "ENABLE_TELEGRAM_NOTIFICATIONS": config.ENABLE_TELEGRAM_NOTIFICATIONS,
    }


def get_cgroup_memory_limit_bytes() -> int:
    """Return the memory limit (bytes) for the current cgroup."""
    # Try cgroup v2 first
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            val = f.read().strip()
            if val != "max":
                return int(val)
    except FileNotFoundError:
        pass
    # Fallback to cgroup v1
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return -1  # Unknown/unlimited

