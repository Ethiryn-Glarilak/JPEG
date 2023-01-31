import cv2
import numpy
import pathlib
from JPEG.JPEG import *

class Image:

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

        self.color = self.image is not None and self.image.ndim == 3

    def RGBtoYUV(self, image):
        # return self.image
        return cv2.cvtColor(image.astype(numpy.float32), cv2.COLOR_RGB2YUV)

    def encode(self, last_value : int = 0, coefficient : int = 50):
        if self.image is None:
            raise ValueError("Image must be provided.")
        self.jpeg = ["", "", ""] if self.color else ""
        DC = [0, 0, 0] if self.color else 0
        for square in self.split_8x8():
            square = self.RGBtoYUV(square) if self.color else square
            if self.color:
                # Ajouter le DC
                value, DC[0] = encode(square[:, :, 0], DC[0], coefficient)
                self.jpeg[0] += value
                value, DC[1] = encode_color(square[:, :, 1], DC[1], coefficient)
                self.jpeg[1] += value
                value, DC[2] = encode_color(square[:, :, 2], DC[2], coefficient)
                self.jpeg[2] += value
            else:
                value, DC = encode(square, last_value, coefficient)
                self.jpeg += value
        return self

    def YUVtoRGB(self, image):
        return cv2.cvtColor(image.astype(numpy.float32), cv2.COLOR_YUV2RGB)

    def decode(self, start : int = 0, last_value : int = 0, coefficient : int = 50):
        shape = self.shape
        if self.jpeg is None:
            raise ValueError("JPEG must be provided.")
        jpeg = True
        start = [start, start, start] if self.color else start
        image = [[], [], []] if self.color else []
        DC = [0, 0, 0] if self.color else 0
        while jpeg:
            if self.color:
                value, start[0], DC[0] = decode(self.jpeg[0], start[0], DC[0], coefficient)
                image[0].append(value)
                value, start[1], DC[1] = decode_color(self.jpeg[1], start[1], DC[1], coefficient)
                image[1].append(value)
                value, start[2], DC[2] = decode_color(self.jpeg[2], start[2], DC[2], coefficient)
                image[2].append(value)
                if start[0] >= len(self.jpeg[0]) or start[1] >= len(self.jpeg[1]) or start[2] >= len(self.jpeg[2]):
                    jpeg = False
            else:
                value, start, DC = decode(self.jpeg, start, last_value, coefficient)
                image.append(value)
                if start >= len(self.jpeg):
                    jpeg = False
        if self.color:
            image[0] = self.concat_8x8(image[0], shape)
            image[1] = self.concat_8x8(image[1], shape)
            image[2] = self.concat_8x8(image[2], shape)
            self.image_decode = self.YUVtoRGB(numpy.moveaxis(numpy.array(image), 0, -1))
        else:
            self.image_decode = self.concat_8x8(image, shape)
        return self

    def split_8x8(self):
        # extend image to shape multiple of 8
        shape = self.image.shape
        if shape[0] % 8 != 0:
            if self.color:
                self.image = numpy.concatenate((self.image, numpy.zeros((8 - shape[0] % 8, shape[1], shape[2]))), axis = 0)
            else:
                self.image = numpy.concatenate((self.image, numpy.zeros((8 - shape[0] % 8, shape[1]))), axis = 0)
        shape = self.image.shape
        if shape[1] % 8 != 0:
            if self.color:
                self.image = numpy.concatenate((self.image, numpy.zeros((shape[0], 8 - shape[1] % 8, shape[2]))), axis = 1)
            else:
                self.image = numpy.concatenate((self.image, numpy.zeros((shape[0], 8 - shape[1] % 8))), axis = 1)
        self.shape = self.image.shape
        # split image into 8x8 squares
        for i in range(0, self.image.shape[0], 8):
            for j in range(0, self.image.shape[1], 8):
                yield self.image[i:i+8, j:j+8]

    def concat_8x8(self, jpeg : list[numpy.ndarray], shape : numpy.ndarray) -> numpy.ndarray:
        # concatenate jpeg into image nxm
        image = jpeg[0]
        for j in range(1, shape[1] // 8):
            image = numpy.concatenate((image, jpeg[j]), axis = 1)
        for i in range(1, shape[0] // 8):
            temp = jpeg[i * shape[0] // 8]
            for j in range(1, shape[1] // 8):
                temp = numpy.concatenate((temp, jpeg[i * shape[0] // 8 + j]), axis = 1)
            image = numpy.concatenate((image, temp), axis = 0)
        return image
