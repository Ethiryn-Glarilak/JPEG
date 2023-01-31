import cv2
import numpy
import matplotlib.pyplot as plt
import pathlib

class DCT:

    def __init__(self,
            path : pathlib.Path = None,
            image : numpy.ndarray = None,
            jpeg : str = None,
    ) -> None:
        if path is not None:
            self.path = pathlib.Path(path)
            if self.path.suffix != ".jpg":
                self.image = cv2.imread(self.path.as_posix())
                self.jpeg = None
            else:
                self.image = None
                self.jpeg = cv2.imread(self.path.as_posix())
        elif image is not None:
            self.image = image
            self.path = None
            self.jpeg = None
        elif jpeg is not None:
            self.jpeg = jpeg
            self.path = None
            self.image = None
        else:
            raise ValueError("Either path or image must be provided.")

    def __getattribute__(self, __name: str) -> None:
        """
        Truc chelou qui fait que d8 se comporte presque comme une variable
        ;)
        """
        methods = ["d8", "JD", "DCT"]
        if __name in methods:
            return super().__getattribute__(__name)()
        return super().__getattribute__(__name)

    def d8(self) -> numpy.ndarray:
        """
        Return matrix d8
        """
        def d8_value(x, y):
            alpha = (x == 0).astype(int)
            alpha[alpha == 0] = 2
            alpha = numpy.sqrt(alpha / 8)
            return numpy.around(alpha * numpy.cos((numpy.pi * (2 * y + 1) * x) / (2 * 8)), 4)
        return numpy.fromfunction(d8_value, shape = (8, 8))

    def JD(self, around : int = 2) -> numpy.ndarray:
        """
        Return matrix JD
        """
        if self.image is None:
            raise ValueError("Image is not loaded")
        JD_calcul = self.d8.dot(self.image).dot(numpy.linalg.inv(self.d8))
        return numpy.around(JD_calcul, around)

    def DCT(self) -> numpy.ndarray:
        """
        Return matrix DCT
        """
        self.image_init = self.image.copy()
        self.image -= 128
        return self.JD

    # --- Method use to show the difference state of compression --- #
    def case(self, m, n):
        """
        Return image with case m, n
        """
        x = numpy.repeat([self.d8[m]], 8, axis = 0).T
        y = numpy.repeat([self.d8[n]], 8, axis = 0)
        return x * y

    def show_state(self, image, state = 1):
        """
        Show the state of compression
        """

        def image_state(state : int) -> numpy.ndarray:
            """
            Calculates the image at state __state__
            """
            image_partiel = numpy.zeros((8, 8))
            JD_copy = numpy.moveaxis(numpy.abs(self.JD.copy()), 0, 1)
            replace_min = numpy.min(JD_copy) - 1

            for _ in range(state):
                index = numpy.where(JD_copy == numpy.max(JD_copy))
                y, x = index[0][0], index[1][0]
                image_partiel += self.case(x, y) * self.JD[x][y]
                JD_copy[y][x] = replace_min

            return numpy.around(image_partiel)

        # Show image, image partiel, and difference image and image partiel #
        image_partiel = image_state(state)
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap = "gray", vmin = 0, vmax = 255)
        plt.subplot(1, 3, 2)
        plt.imshow(image_partiel, cmap = "gray", vmin = 0, vmax = 255)
        plt.subplot(1, 3, 3)
        plt.imshow(image - image_partiel, cmap = "gray", vmin = -255, vmax = 255)
        print(image - image_partiel)
        plt.show()

    def show_courbe(self):

        def root_mean_square_error(image):
            return numpy.sqrt((image ** 2).sum() / (image.shape[0] * image.shape[1]))

        def signal_to_noise_ratio(default, image):
            return 10 * numpy.log10((default ** 2).sum() / ((image ** 2).sum() + 1))

        x = [k for k in range(64)]
        RSME = [root_mean_square_error(self.image - self.image_state(k)) for k in x]
        SNR = [signal_to_noise_ratio(self.image, self.image - self.image_state(k)) for k in x]

        for i in (6, 16, 20, 31, 32):
            print(numpy.around(RSME[i], 1), numpy.around(SNR[i], 1))

        plt.plot(x, RSME, SNR)
        plt.axis([0, 63, 0, 90])
        plt.show()


class JPEG(DCT):

    def coefficient(self, value : int = 50) -> int:
        """
        Return coefficient corresponding to value
        """
        if value > 100 or value < 1:
            value = 50
            print("Value must be between 1 and 100")
        if value < 50:
            alpha = 5000 / value
        else:
            alpha = 200 - 2 * value

        return ((numpy.array([
            [16, 11, 10, 16,  24,  40,  51,  61,],
            [12, 12, 14, 19,  26,  58,  60,  55,],
            [14, 13, 16, 24,  40,  57,  69,  56,],
            [14, 17, 22, 29,  51,  87,  80,  62,],
            [18, 22, 37, 56,  68, 109, 103,  77,],
            [24, 35, 55, 64,  81, 104, 113,  92,],
            [49, 64, 78, 87, 103, 121, 120, 101,],
            [72, 92, 95, 98, 112, 100, 103,  99,],
        ]) * alpha + 50) / 100).astype(int)

    def quantized(self, image, coefficient : numpy.ndarray) -> numpy.ndarray:
        return image/coefficient

    def zigzag_ordering(self, image : numpy.ndarray) -> list[int]:
        def sort(position):
            x, y = position
            return (x + y, -y if (x + y) % 2 else y)
        index = range(self.image.shape[0])
        zigzag = []
        last = 0
        for index, (x, y) in enumerate(sorted(((x, y) for x in index for y in index), key = sort)):
            zigzag.append(image[x][y])
            if self.image[x][y] != 0:
                last = index
        return zigzag[:last + 1] + ["EOB"]

    def DC(self, value : int) -> str:
        if value < 0:
            binary = format(-value, "b").replace("1", "2").replace("0", "1").replace("2", "0")
        else:
            binary = format(value, "b")
        return ["010", "011", "100", "00", "101", "110", "1110", "11110", "111110", "1111110", "11111110", "111111110",][len(binary)] + binary

    def AC(self, value : int, run : int = 0) -> str:
        if value == "EOB":
            return "1010"
        if value < 0:
            binary = format(-value, "b").replace("1", "2").replace("0", "1").replace("2", "0")
        else:
            binary = format(value, "b")
        return [
            [          "00",               "01",              "100",             "1011",            "11010",           "111000",          "1111000",       "1111110110", "1111111110000010", "1111111110000011",],
            [        "1100",           "111001",          "1111001",        "111110110",      "11111110110", "1111111110000100", "1111111110000101", "1111111110000110", "1111111110000111", "1111111110001000",],
            [       "11011",         "11111000",       "1111110111", "1111111110001001", "1111111110001010", "1111111110001011", "1111111110001100", "1111111110001101", "1111111110001110", "1111111110001111",],
            [      "111010",        "111110111",       "1111110111", "1111111110010000", "1111111110010001", "1111111110010010", "1111111110010011", "1111111110010100", "1111111110010101", "1111111110010110",],
            [      "111011",       "1111111000", "1111111110010111", "1111111110011000", "1111111110011001", "1111111110011010", "1111111110011011", "1111111110011100", "1111111110011101", "1111111110011110",],
            [     "1111010",       "1111111001", "1111111110011111", "1111111110100000", "1111111110100001", "1111111110100010", "1111111110100011", "1111111110100100", "1111111110100101", "1111111110100110",],
            [     "1111011",      "11111111000", "1111111110100111", "1111111110101000", "1111111110101001", "1111111110101010", "1111111110101011", "1111111110101100", "1111111110101101", "1111111110101110",],
            [    "11111001",      "11111111001", "1111111110101111", "1111111110110000", "1111111110110001", "1111111110110010", "1111111110110011", "1111111110110100", "1111111110110101", "1111111110110110",],
            [    "11111010",  "111111111000000", "1111111110110111", "1111111110111000", "1111111110111001", "1111111110111010", "1111111110111011", "1111111110111100", "1111111110111101", "1111111110111110",],
            [   "111111000", "1111111110111111", "1111111111000000", "1111111111000001", "1111111111000010", "1111111111000011", "1111111111000100", "1111111111000101", "1111111111000110", "1111111111000111",],
            [   "111111001", "1111111111001000", "1111111111001001", "1111111111001010", "1111111111001011", "1111111111001100", "1111111111001101", "1111111111001110", "1111111111001111", "1111111111010000",],
            [   "111111010", "1111111111010001", "1111111111000010", "1111111111010011", "1111111111010100", "1111111111010101", "1111111111010110", "1111111111010111", "1111111111011000", "1111111111011001",],
            [  "1111111010", "1111111111011010", "1111111111011011", "1111111111011100", "1111111111011101", "1111111111011110", "1111111111011111", "1111111111100000", "1111111111100001", "1111111111100010",],
            ["111111110110", "1111111111101100", "1111111111101101", "1111111111101110", "1111111111101111", "1111111111110000", "1111111111110001", "1111111111110010", "1111111111110011", "1111111111110100",],
            ["111111110111", "1111111111110101", "1111111111110110", "1111111111110111", "1111111111111000", "1111111111111001", "1111111111111010", "1111111111111100", "1111111111111101", "1111111111111110",],
        ][run][len(binary) - 1] + binary

    def encode(self, last_value = 0, value : int = 50) -> str:
        """
        Function to encode image in JPEG format.
        """

        # Create image #
        image = self.quantized(self.DCT, self.coefficient(value))
        image = numpy.around(image, 0).astype(int)
        image = self.zigzag_ordering(image)

        # Convert image
        jpeg = self.DC(image[0] - last_value)
        count_0 = 0
        for k in image[1:]:
            if k == 0:
                count_0 += 1
            else:
                jpeg += self.AC(k, count_0)
                count_0 = 0
        return jpeg

    def extractDC(self, index = 1, last_value : int = 0) -> tuple[list[int], int]:
        DC_code = {
            "010"       : 0,  "011"       : 1,
            "100"       : 2,  "00"        : 3,
            "101"       : 4,  "110"       : 5,
            "1110"      : 6,  "11110"     : 7,
            "111110"    : 8,  "1111110"   : 9,
            "11111110"  : 10, "111111110" : 11,
        }
        while DC_code.get(self.jpeg[:index]) is None:
            index += 1
        index_end = index + DC_code.get(self.jpeg[:index])
        if self.jpeg[index] == "0":
            return [- int(self.jpeg[index:index_end].replace("1", "2").replace("0", "1").replace("2", "0"), 2) + last_value], index_end
        else:
            return [int(self.jpeg[index:index_end], 2) + last_value], index_end

    def extractAC(self, index : int = 0) -> list[int]:
        AC_code = {
            "1010"             : "EOB",
            "00"               : (0, 1), "01"               : (0, 2), "100"              : (0, 3), "1011"             : (0, 4), "11010"            : (0, 5),
            "111000"           : (0, 6), "1111000"          : (0, 7), "1111110110"       : (0, 8), "1111111110000010" : (0, 9), "1111111110000011" : (0, 10),
            "1100"             : (1, 1), "111001"           : (1, 2), "1111001"          : (1, 3), "111110110"        : (1, 4), "11111110110"      : (1, 5),
            "1111111110000100" : (1, 6), "1111111110000101" : (1, 7), "1111111110000110" : (1, 8), "1111111110000111" : (1, 9), "1111111110001000" : (1, 10),
            "11011"            : (2, 1), "11111000"         : (2, 2), "1111110111"       : (2, 3), "1111111110001001" : (2, 4), "1111111110001010" : (2, 5),
            "1111111110001011" : (2, 6), "1111111110001100" : (2, 7), "1111111110001101" : (2, 8), "1111111110001110" : (2, 9), "1111111110001111" : (2, 10),
            "111010"           : (3, 1), "111110111"        : (3, 2), "1111110111"       : (3, 3), "1111111110010000" : (3, 4), "1111111110010001" : (3, 5),
            "1111111110010010" : (3, 6), "1111111110010011" : (3, 7), "1111111110010100" : (3, 8), "1111111110010101" : (3, 9), "1111111110010110" : (3, 10),
            "111011"           : (4, 1), "1111111000"       : (4, 2), "1111111110010111" : (4, 3), "1111111110011000" : (4, 4), "1111111110011001" : (4, 5),
            "1111111110011010" : (4, 6), "1111111110011011" : (4, 7), "1111111110011100" : (4, 8), "1111111110011101" : (4, 9), "1111111110011110" : (4, 10),
            "1111010"          : (5, 1), "1111111001"       : (5, 2), "1111111110011111" : (5, 3), "1111111110100000" : (5, 4), "1111111110100001" : (5, 5),
            "1111111110100010" : (5, 6), "1111111110100011" : (5, 7), "1111111110100100" : (5, 8), "1111111110100101" : (5, 9), "1111111110100110" : (5, 10),
            "1111011"          : (6, 1), "11111111000"      : (6, 2), "1111111110100111" : (6, 3), "1111111110101000" : (6, 4), "1111111110101001" : (6, 5),
            "1111111110101010" : (6, 6), "1111111110101011" : (6, 7), "1111111110101100" : (6, 8), "1111111110101101" : (6, 9), "1111111110101110" : (6, 10),
            "11111001"         : (7, 1), "11111111001"      : (7, 2), "1111111110101111" : (7, 3), "1111111110110000" : (7, 4), "1111111110110001" : (7, 5),
            "1111111110110010" : (7, 6), "1111111110110011" : (7, 7), "1111111110110100" : (7, 8), "1111111110110101" : (7, 9), "1111111110110110" : (7, 10),
            "11111010"         : (8, 1), "111111111000000"  : (8, 2), "1111111110110111" : (8, 3), "1111111110111000" : (8, 4), "1111111110111001" : (8, 5),
            "1111111110111010" : (8, 6), "1111111110111011" : (8, 7), "1111111110111100" : (8, 8), "1111111110111101" : (8, 9), "1111111110111110" : (8, 10),
            "111111000"        : (9, 1), "1111111110111111" : (9, 2), "1111111111000000" : (9, 3), "1111111111000001" : (9, 4), "1111111111000010" : (9, 5),
            "1111111111000011" : (9, 6), "1111111111000100" : (9, 7), "1111111111000101" : (9, 8), "1111111111000110" : (9, 9), "1111111111000111" : (9, 10),
            "111111001"        : (10, 1), "1111111111001000" : (10, 2), "1111111111001001" : (10, 3), "1111111111001010" : (10, 4), "1111111111001011" : (10, 5),
            "1111111111001100" : (10, 6), "1111111111001101" : (10, 7), "1111111111001110" : (10, 8), "1111111111001111" : (10, 9), "1111111111010000" : (10, 10),
            "111111010"        : (11, 1), "1111111111010001" : (11, 2), "1111111111000010" : (11, 3), "1111111111010011" : (11, 4), "1111111111010100" : (11, 5),
            "1111111111010101" : (11, 6), "1111111111010110" : (11, 7), "1111111111010111" : (11, 8), "1111111111011000" : (11, 9), "1111111111011001" : (11, 10),
            "1111111010"       : (12, 1), "1111111111011010" : (12, 2), "1111111111011011" : (12, 3), "1111111111011100" : (12, 4), "1111111111011101" : (12, 5),
            "1111111111011110" : (12, 6), "1111111111011111" : (12, 7), "1111111111100000" : (12, 8), "1111111111100001" : (12, 9), "1111111111100010" : (12, 10),
            "11111111010"      : (13, 1), "1111111111100011" : (13, 2), "1111111111100100" : (13, 3), "1111111111100101" : (13, 4), "1111111111100110" : (13, 5),
            "1111111111101111" : (13, 6), "1111111111101000" : (13, 7), "1111111111101001" : (13, 8), "1111111111101010" : (13, 9), "1111111111101011" : (13, 10),
            "111111110110"     : (14, 1), "1111111111101100" : (14, 2), "1111111111101101" : (14, 3), "1111111111101110" : (14, 4), "1111111111101111" : (14, 5),
            "1111111111110000" : (14, 6), "1111111111110001" : (14, 7), "1111111111110010" : (14, 8), "1111111111110011" : (14, 9), "1111111111110100" : (14, 10),
            "111111110111"     : (15, 1), "1111111111110101" : (15, 2), "1111111111110110" : (15, 3), "1111111111110111" : (15, 4), "1111111111111000" : (15, 5),
            "1111111111111001" : (15, 6), "1111111111111010" : (15, 7), "1111111111111100" : (15, 8), "1111111111111101" : (15, 9), "1111111111111110" : (15, 10),
        }
        AC = []
        i = index
        while True:
            i += 1
            if AC_code.get(self.jpeg[index:i]) is None:
                continue
            if AC_code.get(self.jpeg[index:i]) == "EOB":
                AC.append(AC_code.get(self.jpeg[index:i]))
                return AC, i
            index_end = i + AC_code.get(self.jpeg[index:i])[1]
            for _ in range(AC_code.get(self.jpeg[index:i])[0]):
                AC.append(0)
            if self.jpeg[i] == "0":
                AC.append(- int(self.jpeg[i:index_end].replace("1", "2").replace("0", "1").replace("2", "0"), 2))
            else:
                AC.append(int(self.jpeg[i:index_end], 2))
            i = index_end
            index = index_end

    def list_to_zigzag(self, zigzag : list[int]) -> numpy.ndarray:
        def sort(position):
            x, y = position
            return (x + y, -y if (x + y) % 2 else y)
        index = range(8)
        image = numpy.zeros((8, 8))
        for index, (x, y) in enumerate(sorted(((x, y) for x in index for y in index), key = sort)):
            if zigzag[index] == "EOB":
                break
            image[x][y] = zigzag[index]
        return image

    def JDm1_calcul(self, image : numpy.ndarray, around : int = 2) -> numpy.ndarray:
        return numpy.around(numpy.linalg.inv(self.d8).dot(image).dot(self.d8), around)

    def decode(self, start = 0, last_value : int = 0, value : int = 50) -> numpy.ndarray:
        image, index = self.extractDC(start, last_value)
        extend, index = self.extractAC(index)
        image.extend(extend)
        image = self.list_to_zigzag(image) * self.coefficient(value)
        image = self.JDm1_calcul(image)
        return image + 128, index

    # --- Method use to show the difference state of compression --- #
    def show_image_after_decompress(self, last_value : int = 0):
        image = self.image.copy()
        image_encode = self.encode(last_value)
        self.jpeg = image_encode
        image_decode, index = self.decode(last_value)
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap = "gray", vmin = 0, vmax = 255)
        plt.subplot(1, 2, 2)
        plt.imshow(image_decode, cmap = "gray", vmin = 0, vmax = 255)
        plt.show()

# JPEG(pathlib.Path("image_chat.png"))

def image() -> numpy.ndarray:
    """Return image"""
    return numpy.array([
        [52, 55, 61,  66,  70,  61, 64, 73,],
        [63, 59, 55,  90, 109,  85, 69, 72,],
        [62, 59, 68, 113, 144, 104, 66, 73,],
        [63, 58, 71, 122, 154, 106, 70, 69,],
        [67, 61, 68, 104, 126,  88, 68, 70,],
        [79, 65, 60,  70,  77,  68, 58, 75,],
        [85, 71, 64,  59,  55,  61, 65, 83,],
        [87, 79, 69,  68,  65,  76, 78, 94,],
    ])

JPEG(image = image()).show_image_after_decompress(-33)
