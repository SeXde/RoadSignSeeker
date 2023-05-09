import os

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_image(path: str) -> []:
    return cv2.imread(path, cv2.IMREAD_ANYCOLOR)


def load_images_from_folder(folder: str) -> []:
    images = []
    for filename in os.listdir(folder):
        img = load_image(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

    return images


def show_image(image, title = ''):
    cv2.imshow(title, image)
    cv2.waitKey(0)


def find_contours(image, class_check: int, C: [], E: []) -> []:
    mask = cv2.adaptiveThreshold(image, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY, 5, -50)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Sacadita

    mask = cv2.dilate(mask, kernel)

    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contour:
        x, y, w, h = cv2.boundingRect(cont)

        max_size = h if h > w else w
        min_size = w if h > w else h

        diff = max_size - min_size
        if diff % 2 != 0:
            max_size += 1
            diff += 1

        im_rect = image[y:y + h, x:x + w]

        v_pad = (0,) if h > w else (diff // 2,)
        h_pad = (diff // 2,) if h > w else (0,)

        padded_im = np.pad(im_rect, [v_pad, h_pad], mode='constant', constant_values=(255, 255))

        resized_im = cv2.resize(padded_im, (25, 25))

        c = np.array(resized_im.flatten())

        C.append(c)
        E.append(class_check)


def test_with(file: str) -> int:
    test_image = load_image(f"resources/{file}")

    train_C = []

    find_contours(test_image, 0, train_C, [])

    XR = lda.transform(train_C)

    XR_float32 = np.array(XR, dtype=np.float32)

    result = bayes.predict(XR_float32)

    return result[1][0][0]


folders = [os.path.join("resources/train_ocr/", str(i)) for i in range(10)]

E = []
C = []
class_check = 0

for folder in folders:
    print(f"Loading images in folder: {folder}")
    class_check += 1

    images = load_images_from_folder(folder)

    for im in images:
        find_contours(im, class_check, C, E)


lda = LinearDiscriminantAnalysis()
lda.fit(C, E)
CR = lda.transform(C)

bayes = cv2.ml.NormalBayesClassifier_create()

CR_float = np.array(CR, dtype=np.float32)
E_int32 = np.array(E, dtype=np.int32)

bayes.train(CR_float, cv2.ml.ROW_SAMPLE, E_int32)

test_images = [
    "validation_ocr/1/0249.png",
    "validation_ocr/3/0200.png",
    "validation_ocr/5/0050.png",
    "validation_ocr/8/0249.png",
]

for test_image in test_images:
    result = test_with(test_image)
    print(f"Image: {test_image}, Result: {result - 1}")


# opencv bayes.fit(CR, E)
#                  ^ float32
#                       ^ int32