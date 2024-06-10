import torch
import torchvision.transforms as transforms

def get_transforms(train=True):
    """
    Returns a dictionary containing the training and validation transforms.
    """
    # Define the transforms for pre-processing the data
    transform = [
        transforms.Resize((256, 256), antialias=True),
        transforms.RandomCrop(224) if train else transforms.CenterCrop(224),
    ]

    if train:
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.RandomRotation(degrees=15))
        transform.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
        transform.append(transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15))

    transform.append(transforms.ToTensor())

    transform.append(
        transforms.Normalize(
            mean=[0.5984658002853394, 0.3079698085784912, 0.24672217667102814], 
            std=[0.19015902280807495, 0.14778202772140503, 0.11925168335437775]
        )
    )
    
    return transforms.Compose(transform)


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
