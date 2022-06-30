import numpy
import matplotlib.pyplot as plt
from JPEG.JPEG.encode.function import *
from JPEG.JPEG.decode.function import *


def show_image_after_decompress(image : numpy.ndarray, last_value : int = 0):
    image_encode = encode(image, last_value)
    jpeg = image_encode
    image_decode, index = decode(jpeg, last_value)
    plt.imshow(image_decode, cmap = "gray", vmin = 0, vmax = 255)
    plt.show()
