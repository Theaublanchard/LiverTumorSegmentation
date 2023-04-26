import numpy as np
from torchvision import transforms

class GeomTransform():
    ''' Base class for geometric transformations that need to be applied to all
        images, masks included.'''
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return x
    
    def update_params(self):
        pass


class RandomRotationOneShot(GeomTransform):
    def __init__(self, angle, p=0.5):	
        self.angle = angle
        self.p = p
        self.rotate = None
        self.rotate_angle = None

    def __call__(self, x):
        if self.rotate:
            x = transforms.functional.rotate(x, self.rotate_angle)
        return x

    def update_params(self):
        self.rotate = np.random.uniform() < self.p
        self.rotate_angle = np.random.uniform(-self.angle, self.angle)


class RandomHorizontalFlipOneShot(GeomTransform):
    def __init__(self, p=0.5):
        self.p = p
        self.flip = None

    def __call__(self, x):
        if self.flip:
            x = transforms.functional.hflip(x)
        return x

    def update_params(self):
        self.flip = np.random.uniform() < self.p


class RandomVerticalFlipOneShot(GeomTransform):
    def __init__(self, p=0.5):
        self.p = p
        self.flip = None

    def __call__(self, x):
        if self.flip:
            x = transforms.functional.vflip(x)
        return x

    def update_params(self):
        self.flip = np.random.uniform() < self.p



mean_ct = np.load('./data/ct_mean.npy')
std_ct = np.load('./data/ct_std.npy')


ct_transform = transforms.Compose([transforms.Normalize(mean_ct, std_ct),
                                #    transforms.RandomApply([transforms.GaussianBlur(kernel_size = 5, sigma=(0.1, 2.0))], p=0.5),
                                ])
aug_transform = [
                #  RandomHorizontalFlipOneShot(p=0.5),
                #  RandomVerticalFlipOneShot(p=0.5),
                 RandomRotationOneShot(15, p=0.5),
                 transforms.RandomApply([transforms.GaussianBlur(kernel_size = 5, sigma=(0.1, 2.0))], p=0.5),

                 ]
