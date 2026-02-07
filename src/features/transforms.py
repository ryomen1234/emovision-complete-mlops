from torchvision import transforms

def get_transform(train: bool = True):

    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                *base_transforms,
            ]
        )
    else:
        return transforms.Compose(base_transforms)
