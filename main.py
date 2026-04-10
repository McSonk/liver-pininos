# This script is to be run as an alternative to the jupyter notebook.
from idssp.sonk.model.data import DataWrapper
from idssp.sonk.model.training import ModelBuilder

if __name__ == "__main__":
    print("Initializing data wrapper and model builder...")
    wrapper = DataWrapper()
    VOLUMES_TO_ANALYSE = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Get the file paths
    paths = []
    train_files = []
    val_files = []

    for volume in VOLUMES_TO_ANALYSE:
        path = wrapper.get_paths_of_volume(volume)
        paths.append(path)


    train_files.extend(paths[:8])
    val_files.extend(paths[8:])

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
