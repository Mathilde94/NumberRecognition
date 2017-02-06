import matplotlib.pyplot as plt

from .exceptions import NoImagesToDisplay


def show(images=[]):

    if not images:
        raise NoImagesToDisplay()

    n = len(images)
    fig = plt.figure()

    for index, image in enumerate(images):
        new_image = fig.add_subplot(1, n, index+1)
        new_image.set_title(image.title)
        plt.imshow(image.content, cmap=plt.cm.gray)
    plt.show()
