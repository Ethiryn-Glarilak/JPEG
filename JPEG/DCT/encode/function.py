import numpy
from JPEG.DCT.function import d8_calcul

def JD(image : numpy.ndarray, d8 = d8_calcul(), around : int = 2) -> numpy.ndarray:
    """
    Return matrix JD
    """
    if image is None:
        raise ValueError("Image is not loaded")
    JD_calcul = d8.dot(image).dot(numpy.linalg.inv(d8))
    return numpy.around(JD_calcul, around)

def DCT(image : numpy.ndarray) -> numpy.ndarray:
    """
    Return matrix DCT
    """
    return JD(image - 128)
