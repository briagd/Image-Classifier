from torchvision import transforms


class Transformations:
    def __init__(self, input_size):

        self.training = transforms.Compose(
            [
                transforms.Resize([input_size, input_size]),
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=15,
                    translate=None,
                    scale=None,
                    shear=15,
                    resample=0,
                    fillcolor=0,
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.001, 2.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.validation = transforms.Compose(
            [
                transforms.Resize([input_size, input_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
