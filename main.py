# This script is to be run as an alternative to the jupyter notebook.
from idssp.sonk.model.data import DataWrapper
from idssp.sonk.model.training import ModelBuilder

if __name__ == "__main__":
    print("Initializing data wrapper and model builder...")
    wrapper = DataWrapper()
    VOLUMES_TO_ANALYSE = [2, 3]

    # Get the file paths
    train_files = []
    val_files = []

    train_files.append(wrapper.get_paths_of_volume(VOLUMES_TO_ANALYSE[0]))
    val_files.append(wrapper.get_paths_of_volume(VOLUMES_TO_ANALYSE[1]))

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
