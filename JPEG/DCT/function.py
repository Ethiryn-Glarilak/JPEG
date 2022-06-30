import numpy

def d8_calcul() -> numpy.ndarray:
    """
    Return matrix d8
    """
    def d8_value(x, y):
        alpha = (x == 0).astype(int)
        alpha[alpha == 0] = 2
        alpha = numpy.sqrt(alpha / 8)
        return numpy.around(alpha * numpy.cos((numpy.pi * (2 * y + 1) * x) / (2 * 8)), 4)
    return numpy.fromfunction(d8_value, shape = (8, 8))
