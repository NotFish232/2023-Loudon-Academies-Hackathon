import torch as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import Self
from typing import Callable
import os


class PirateDataset(Dataset):
    def __init__(
        self: Self, img_dir: str = "imgs", transforms: Callable = None
    ) -> None:
        self.img_dir = img_dir
        self.imgs = [img for img in os.listdir(img_dir) if img.endswith(".png")]
        self.transforms = transforms

    def __len__(self: Self) -> int:
        return len(self.imgs)

    def __getitem__(self: Self, idx: int) -> T.Tensor:
        img_path = f"{self.img_dir}/{self.imgs[idx]}"
        img = Image.open(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img


def main() -> None:
    dataset = PirateDataset(transforms=transforms.ToTensor())
    print(dataset[0].shape)


if __name__ == "__main__":
    main()
