import random
import torch

from torchvision.transforms import functional as F
#import torch.nn.functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class Adjust_contrast(object):

    def __call__(self, image, target):
        image= F.to_pil_image(image)
        image= F.adjust_contrast(image,2)
        image= F.to_tensor(image)
        return image, target

class Adjust_brightness(object):

    def __call__(self, image, target):
        image= F.to_pil_image(image)
        image= F.adjust_brightness(image, 2)
        image= F.to_tensor(image)
        return image, target

class Adjust_saturation(object):
    def __call__(self, image, target):
        image= F.to_pil_image(image)
        image= F.adjust_saturation(image, 3)
        image= F.to_tensor(image)
        return image, target

class lighting_noise(object):

    def __call__(self, image, target):
        new_image= F.to_pil_image(image)
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
             (1, 2, 0), (2, 0, 1), (2, 1, 0))
        swap = perms[random.randint(0, len(perms)- 1)]
        new_image = F.to_tensor(new_image)
        new_image = new_image[swap, :, :]
        return image, target

class adjust_hue(object):

    def __call__(self, image, target):
        image= F.to_pil_image(image)
        image= F.adjust_hue(image, 0.5)
        image= F.to_tensor(image)
        return image, target
class adjust_gamma(object):

    def __call__(self, image, target):
        image= F.to_pil_image(image)
        image= F.adjust_gamma(image, 3)
        image= F.to_tensor(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
