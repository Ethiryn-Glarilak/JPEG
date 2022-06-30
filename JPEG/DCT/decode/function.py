import numpy
from JPEG.DCT.function import d8_calcul

def JDm1_calcul(image : numpy.ndarray, around : int = 2, d8 = d8_calcul()) -> numpy.ndarray:
    return numpy.around(numpy.linalg.inv(d8).dot(image).dot(d8), around)
