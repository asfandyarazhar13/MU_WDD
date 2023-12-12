class CustomDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        """
        Custom dataset initialization.
        :param images_file: Path to the .pt file containing images.
        :param labels_file: Path to the .pt file containing labels.
        :param transform: Transformations to be applied on images.
        """
        self.images = torch.load(images_file)
        self.labels = torch.load(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def cifar10_custom(batch_size=128, data_dir="datasets/cifar10", num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the custom dataset
    custom_train_set = CustomDataset(
        images_file='path/to/image.pt', 
        labels_file='path/to/labels.pt', 
        transform=train_transform
    )

    val_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    # Create data loaders
    custom_train_loader = DataLoader(
        custom_train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return custom_train_loader, val_loader, test_loader

# Usage
train_loader, val_loader, test_loader = cifar10_dataloaders_no_val()
