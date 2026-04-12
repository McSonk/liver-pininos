from idssp.sonk import config
from monai.utils import set_determinism

from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.data import DataWrapper
from idssp.sonk.model.training import ModelBuilder

# For reproducibility
set_determinism(seed=42)

if __name__ == "__main__":
    print("Reading directories...")
    loader = DataCollector()
    loader.read_dir(config.CT_ROOT, ds_source='LiTS')
    loader.extract_images_and_labels()
    print("splitting data into train and val sets...")
    train_files, val_files = loader.get_reproducible_split()
    print("Initializing data wrapper and model builder...")
    wrapper = DataWrapper()

    if config.is_limited_env():
        print("Limited environment detected. Using a subset of the data for quick testing.")
        train_files = train_files[:2]  # Use only 2 samples for training
        val_files = val_files[:2]      # Use only 2 samples for validation

    print("Training files:")
    for file in train_files:
        print(file)

    print("Validation files:")
    for file in val_files:
        print(file)

    builder = ModelBuilder()
    builder.init_data_loaders(train_files, val_files)

    builder.init_model()
    print("Model initialized. Starting training...")
    builder.train()
