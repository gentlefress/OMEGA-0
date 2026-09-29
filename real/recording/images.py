"""Episode-local PNG sequences, including uint16 depth in millimeters."""
from collections.abc import Mapping
from pathlib import Path
import re

import numpy as np


def image_directories(layout):
    if layout is None:
        return {}
    if not isinstance(layout, Mapping):
        raise ValueError('layout.images must map directory names to input names')
    result = {}
    for name, stream in layout.items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name)):
            raise ValueError(f'Invalid image directory: {name!r}')
        if not isinstance(stream, str) or not stream:
            raise ValueError(f'Image directory {name!r} must name an input')
        if stream in result.values():
            raise ValueError(f'Image input {stream!r} is mapped more than once')
        result[name] = stream
    return result


def png_array(value):
    """Convert floating depth (mm) to PNG uint16; preserve integer images."""
    image = np.asarray(value)
    if image.ndim == 2 and image.dtype.kind == 'f':
        # ZED marks invalid depth per pixel. PNG has no NaN representation.
        image = np.clip(np.nan_to_num(image, nan=0., posinf=0., neginf=0.), 0, 65535).astype(np.uint16)
    if not (image.dtype == np.uint8 and (image.ndim == 2 or image.ndim == 3 and image.shape[-1] in (3, 4))
            or image.dtype == np.uint16 and image.ndim == 2):
        raise ValueError('PNG images require uint8 gray/RGB/RGBA, uint16 depth, or floating HW depth in millimeters')
    if any(size == 0 for size in image.shape):
        raise ValueError('PNG images cannot be empty')
    return image


def read_image(path):
    from PIL import Image
    with Image.open(path) as image:
        if image.format != 'PNG':
            raise ValueError(f'Expected PNG image: {path}')
        # Older Pillow releases decode 16-bit grayscale PNG into int32 mode I.
        return np.array(image, dtype=np.uint16 if image.mode in {'I', 'I;16', 'I;16B'} else None)


class ImageWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir()
        self.layout = None
        self.count = 0

    def validate(self, image):
        image = png_array(image)
        if self.layout is not None and self.layout != (image.shape, image.dtype):
            raise ValueError('PNG image layout changed during episode')
        return image

    def write(self, image):
        from PIL import Image
        image = self.validate(image)
        index = self.count
        target = self.path / f'frame_{index:06d}.png'
        temporary = target.with_suffix('.png.tmp')
        Image.fromarray(image).save(temporary, format='PNG')
        temporary.replace(target)
        self.layout = image.shape, image.dtype
        self.count += 1
        return index


def sequence_length(directory):
    """Return the contiguous prefix and whether numbered PNGs have gaps."""
    paths = list(directory.glob('frame_*.png'))
    indices = set()
    for path in paths:
        match = re.fullmatch(r'frame_(\d{6,})\.png', path.name)
        if not match or path.name != f'frame_{int(match[1]):06d}.png':
            raise ValueError(f'Invalid image frame filename: {path.name}')
        indices.add(int(match[1]))
    count = 0
    while count in indices:
        count += 1
    return count, count != len(indices)
