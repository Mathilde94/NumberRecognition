import numpy as np
from scipy import misc, ndimage

from .exceptions import NoImageFileFound
from .filters import gaussian_filter


class Image:

    def __init__(self, title='', content=[], file=None, ):
        self.content = content
        self.title = title

        if file:
            self.load_from_file(file)

    def load_from_file(self, input_file=None):
        try:
            self.content = misc.imread(input_file)
        except IOError:
            raise NoImageFileFound
        if not self.title:
            self.title = input_file

    def get_crop_square(self, start=0, size=0):
        content = self.content[start:size, start:size]
        title = "{}--cropped".format(self.title)
        return Image(title, content)

    def grey(self):
        content = np.zeros((self.content.shape[0], self.content.shape[1]))
        for i in range(self.content.shape[0]):
            for j in range(self.content.shape[1]):
                content[i][j] = np.average(self.content[i][j])

        title = "{}--grey".format(self.title)
        return Image(title, content)

    def gaussian(self, sigma=3):
        blurred_content = ndimage.gaussian_filter(self.content, sigma=sigma)
        return Image("{}--gaussian_{}".format(self.title, sigma), blurred_content)

    def custom_gaussian(self, sigma=3):
        blurred_content = gaussian_filter(self.content, sigma=sigma)
        return Image("{}--custom-gaussian_{}".format(self.title, sigma), blurred_content)
