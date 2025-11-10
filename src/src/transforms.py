import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transforms(img_size, mean, std, color_jitter=0.2, randaugment=True):
    aug = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.05, p=0.5),
        A.CoarseDropout(min_holes=1, max_holes=1, min_height=int(0.1*img_size), max_height=int(0.2*img_size), min_width=int(0.1*img_size),  max_width=int(0.2*img_size), p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    return A.Compose(aug)

def val_transforms(img_size, mean, std):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
