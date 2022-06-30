import numpy
import matplotlib.pyplot as plt
from JPEG.DCT import *

# --- Function use to show the difference state of compression --- #
def case(m, n, d8 = d8_calcul()):
    """
    Return image with case m, n
    """
    x = numpy.repeat([d8[m]], 8, axis = 0).T
    y = numpy.repeat([d8[n]], 8, axis = 0)
    return x * y

def image_state(JD : numpy.ndarray, state : int) -> numpy.ndarray:
    """
    Calculates the image at state __state__
    """
    image_partiel = numpy.zeros((8, 8))
    JD_copy = numpy.moveaxis(numpy.abs(JD.copy()), 0, 1)
    replace_min = numpy.min(JD_copy) - 1

    for _ in range(state):
        index = numpy.where(JD_copy == numpy.max(JD_copy))
        y, x = index[0][0], index[1][0]
        image_partiel += case(x, y) * JD[x][y]
        JD_copy[y][x] = replace_min

    return numpy.around(image_partiel)

def show_state(image, state = 1):
    """
    Show the state of compression
    """

    # Show image, image partiel, and difference image and image partiel #
    image_partiel = image_state(JD(image), state)
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap = "gray", vmin = 0, vmax = 255)
    plt.subplot(1, 3, 2)
    plt.imshow(image_partiel, cmap = "gray", vmin = 0, vmax = 255)
    plt.subplot(1, 3, 3)
    plt.imshow(image - image_partiel, cmap = "gray", vmin = -255, vmax = 255)
    print(image - image_partiel)
    plt.show()

def show_courbe(image : numpy.ndarray) -> None:

    def root_mean_square_error(image):
        return numpy.sqrt((image ** 2).sum() / (image.shape[0] * image.shape[1]))

    def signal_to_noise_ratio(default, image):
        return 10 * numpy.log10((default ** 2).sum() / ((image ** 2).sum() + 1))

    x = [k for k in range(64)]
    JD_value = JD(image)
    RSME = [root_mean_square_error(image - image_state(JD_value, k)) for k in x]
    SNR = [signal_to_noise_ratio(image, image - image_state(JD_value, k)) for k in x]

    for i in (6, 16, 20, 31, 32):
        print(numpy.around(RSME[i], 1), numpy.around(SNR[i], 1))

    plt.plot(x, RSME, SNR)
    plt.axis([0, 63, 0, 90])
    plt.show()
