"""Image transforms used by the training pipelines."""
from typing import Tuple
from pydantic import BaseModel, Field
from torchvision.transforms import v2


class ResizeImage(BaseModel):
    size: int | tuple[int, int] = (256, 480)

    def __call__(self):
        return v2.Resize(self.size, interpolation=v2.InterpolationMode.NEAREST)

    @property
    def resolution(self) -> Tuple[int, int]:
        if isinstance(self.size, int):
            return (self.size, self.size)
        elif isinstance(self.size, list) and len(self.size) == 2:
            return (self.size[0], self.size[1])
        elif isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        else:
            raise ValueError('size should be int or list of two ints')


class ColorJitter(BaseModel):
    brightness: float | tuple[float, float] = Field(default_factory=lambda : 0.2)
    contrast: float | tuple[float, float] = Field(default_factory=lambda : (0.8, 1.2))
    saturation: float | tuple[float, float] = Field(default_factory=lambda : (0.8, 1.2))
    hue: float | tuple[float, float] = Field(default_factory=lambda : 0.05)

    def __call__(self):
        return v2.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)


class CenterCrop(BaseModel):
    size: int | tuple[int, int] = (224, 224)

    def __call__(self):
        return v2.CenterCrop(self.size)
