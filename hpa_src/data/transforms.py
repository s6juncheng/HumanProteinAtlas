from skimage import io, transform
import torch
from PIL import Image

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='reflect', anti_aliasing=True)

        return img
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()

class ToNumpy(object):
    """ Convert torch tensor to numpy
    """
    def __call__(self, image):
        # swap color axis because
        # torch image: C X H X W  
        # numpy image: H x W x C
        image = image.permute(1,2,0)
        return image.numpy()
    
class ToPIL(object):
    """ Convert to PIL Image
    """
    def __call__(self, image):
        image = Image.fromarray(image)
        return image