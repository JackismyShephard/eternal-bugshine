def split_dataset(dataset, train_ratio, val_ratio):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * (len(dataset) - train_size))
    test_size = len(dataset) - (train_size + val_size)
    dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}
    train_data, val_data, test_data = random_split(dataset, dataset_sizes.values(),
                                                   generator=torch.manual_seed(RNG_SEED))
    return train_data, val_data, test_data, dataset_sizes


def default_transform(train_data, val_data, test_data, shape=(224, 448),
                      mean=BEETLENET_MEAN, std=BEETLENET_STD):

    resize = transforms.Resize(shape)
    tensorfy = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([resize, tensorfy, normalize])
    train_data_T = TransformsDataset(train_data, transform)
    val_data_T = TransformsDataset(val_data, transform)
    test_data_T = TransformsDataset(test_data, transform)

    return train_data_T, val_data_T, test_data_T
