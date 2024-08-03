import os
import torch
import cv2
import numpy as np
from numpy import ndarray
from torch import Tensor
from typing import Any
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def rebuild_images(root, images):
    rebud_images = []
    for image in images:
        path = os.path.join(root, image[0])
        item = (path, image[1])
        rebud_images.append(item)

    return rebud_images


def getClasses_idxs_Imgs(root):
    classes, class_to_idx = find_classes(root)
    imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)

    return classes, class_to_idx, imgs


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()
    # print( tensor.dtype )
    return tensor


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def pil_loader_swin(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    #     img = Image.open(f)
    #     return img.convert('RGB')

    image = cv2.imread( path ).astype( np.float32 )

    # BGR to RGB
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )

    # Convert image data to pytorch format data
    # tensor = image_to_tensor( image, False, False )#.unsqueeze_( 0 )
    image = Image.fromarray(np.uint8(image))
    return image


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)

    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for _, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(target, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


class MyDataset(data.Dataset):
    def __init__(self, root, transform_s=None, transform_e=None, transform_m=None,
                 loader=pil_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform_s = transform_s
        self.transform_e = transform_e
        self.transform_m = transform_m
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))

        imhr = None
        imlr = None
        if self.transform_s is not None:
            img = self.transform_s(img)  # M*N-> 224*224
            imhr = self.transform_e(img)   # To Tensor
        if self.transform_m is not None:
            imlr = self.transform_m(img)  # 224*224->56*56
            imlr = self.transform_e(imlr)  # To Tensor
        if imlr is None:
            imlr = imhr

        return imhr, imlr, target

    def __len__(self):
        return len(self.imgs)


class MyDatasetSwin(data.Dataset):
    def __init__(self, root, transform_s=None, transform_e=None, transform_m=None,
                 loader=pil_loader_swin):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform_s = transform_s
        self.transform_e = transform_e
        self.transform_m = transform_m
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))
        # print( img.shape )
        imhr = None
        imlr = None
        if self.transform_s is not None:
            img = self.transform_s(img)  # M*N-> 224*224
            # print(img.shape)
            imhr = self.transform_e(img)   # To Tensor
            # print( img.shape )
        if self.transform_m is not None:
            imlr = self.transform_m(img)  # 224*224->56*56
            imlr = self.transform_e(imlr)  # To Tensor
        if imlr is None:
            imlr = imhr

        return imhr, imlr, target

    def __len__(self):
        return len(self.imgs)