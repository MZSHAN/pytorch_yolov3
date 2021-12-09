import pathlib
from torch.utils.data import DataLoader, Dataset

#TODO: Change this function later, it is not pythonic
# Change for distributed training and multiple workers, collate fn etc
def get_data_loader(config_path, config_reader_class, dataset_class, batch_size=1, mode="train"):
    config_reader = config_reader_class(config_path)
    data_config = config_reader.parse_config()

    train_file = pathlib.Path.cwd().parent / data_config[mode]
    dataset = dataset_class(str(train_file))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader