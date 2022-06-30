import cv2
import matplotlib.pyplot as plt
import JPEG
import numpy

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

image_3 = cv2.cvtColor(cv2.imread("./image_chat.png"), cv2.COLOR_BGR2RGB)

# image_2 = (plt.imread("./image_chat.png") * 255).astype(int)

# image_4 = (plt.imread("./image_chat.png", "RGB")[:8, :16] * 255).astype(int)
image = JPEG.Image(image = image_3)
# print(image.image)
print("Encode")
image.encode()
print("Decode")
test = image.decode()
plt.subplot(1, 2, 2)
plt.imshow(test.image_decode.astype(int))
plt.subplot(1, 2, 1)
plt.imshow(image_3)
print("Fin")
plt.show()
# plt.imshow(test.image.astype(int))
# plt.show()
# print(numpy.max(test.image_decode))
# print(numpy.min(test.image_decode))
