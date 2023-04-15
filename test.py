import cv2
import numpy as np
from common.shape import Shape


def get_mask():
    return np.ones((40, 80))


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def aabb_test(a: Shape, b: Shape) -> bool:
    return a.minX <= b.maxX and a.maxX >= b.minX and a.minY <= b.maxY and a.maxY >= b.minY


def show_image(img, title = ''):
    cv2.imshow(title, img)
    cv2.waitKey(0)

# Leer imagen
# Pasar a gris
# Mejorar contraste (Ecualización)
# MSER definir filtros a nivel de create
# (cv2.boundingRect)
# Relacción de aspecto
# Quitar borde blanco
# Máscara color azul saturado HSV 40x80
# Resize de detección a 40x80
# Extraer máscara de color azulado ??????? WTF mama?
# Correlar las mascaritas
# Establecer scoring y tal
# Crear clase para detectar azul (HSV, hue)


img = cv2.imread('resources/test_detection/00014.png', cv2.IMREAD_ANYCOLOR)
Irgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Ihsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
IMarked = img.copy()

IRoadSign_1Mask = cv2.imread('resources/RoadSign_Mask.png', cv2.IMREAD_GRAYSCALE)

Igray = cv2.equalizeHist(Igray)

# plt.figure(figsize=(15,10))
# plt.imshow(Irgb)
# plt.show()

output = np.zeros((Igray.shape[0], Igray.shape[1], 3), dtype=np.uint8)
# mser = cv2.MSER_create(delta=5,max_variation=0.5,max_area=20000) # ORIGINAL
mser = cv2.MSER_create(delta=5, max_variation=0.5, max_area=20000)
polygons = mser.detectRegions(Igray)

print("FOUND", len(polygons[0]))

selected = []
selected_corr = []

idx = 0
for polygon in polygons[0]:
    idx += 1
    # colorRGB = hsv_to_rgb(random(),1,1)
    # colorRGB = tuple(int(color*255) for color in colorRGB)
    # output = cv2.fillPoly(output,[polygon],colorRGB)
    x, y, w, h = cv2.boundingRect(polygon)

    # FILTER Compare by ratio (width / height)
    ratio = w / h
    if w < 50:
        continue

    if ratio < 1.5:
        continue

    # Compare by color (DOMINANT)
    x1 = (x, y)
    x2 = (x, y + h)
    x3 = (x + w, y + h)
    x4 = (x + w, y)

    pixelsInRect = Ihsv[y:y + h, x:x + w]

    dominantColor = unique_count_app(pixelsInRect)
    # print(bb)

    # if dominantColor[2] < 150 or dominantColor[0] > 50:
    #    continue
    if dominantColor[0] < 150 and dominantColor[0] > 250:
        continue

    hsvRect = Ihsv[y:y + h, x:x + w]
    rgbRect = Irgb[y:y + h, x:x + w]

    # lower_white = np.array([0,0,200])
    # upper_white = np.array([255,100,255])

    # print("=====================")
    # print("=====================")
    # print(hsvRect[:,:,0])
    # print("=====================")
    # print("=====================")
    # print(hsvRect[:,:,1])
    # print("=====================")
    # print("=====================")
    # print(hsvRect[:,:,2])
    # print("=====================")
    # print("=====================")

    hsvRectResized = cv2.resize(hsvRect, (80, 40))

    lower_blue = np.array([150, 150, 50])
    upper_blue = np.array([250, 255, 255])

    hsvRectResized = cv2.inRange(hsvRectResized, lower_blue, upper_blue)

    corrArr = cv2.matchTemplate(hsvRectResized, IRoadSign_1Mask, cv2.TM_CCORR_NORMED)
    corr = corrArr[0, 0]
    # corr2 = cv2.matchTemplate(hsvRectResized, np.ones(hsvRectResized.shape, dtype=hsvRectResized.dtype), cv2.TM_CCORR_NORMED)

    threshold = 0.75
    if corr < threshold:
        continue

    # DRAW RESULT
    # cv2.rectangle(IMarked, (x, y), (x + w, y + h), IMarked.shape, 2)

    # print(corr)

    # plt.figure(figsize=(15,10))
    # plt.imshow(rgbRect, cmap="gray")
    # plt.show()
    selected = selected + [Shape(x, x + w, y, y + h)]
    selected_corr = selected_corr + [corr]

    # score = (corr - threshold) / (1 - threshold)

# final_selection = selected.copy()
num_images = len(selected)

# for shape_a_index in range(num_images):
#     shapeA = selected[shape_a_index]
#     is_valid = True
#     shapeB = None
#
#     for shape_b_index in range(len(final_selection)):
#         shapeB = final_selection[shape_b_index]
#
#         if not shape_a_index == shape_b_index and aabb_test(shapeA, shapeB):
#             is_valid = False
#             break

#     if not is_valid:
#         final_selection.remove(shapeB)

# show_image(IMarked)

if len(selected) == 0:
    exit()

classes = np.zeros(num_images)
c = 1

classes[0] = c

for a_index in range(num_images):
    shapeA = selected[a_index]

    for b_index in range(a_index, num_images):
        shapeB = selected[b_index]

        if a_index == b_index:
            continue

        if aabb_test(shapeA, shapeB):
            classes[b_index] = classes[a_index]
        else:
            c = c + 1
            classes[b_index] = c
            break

available_classes = np.unique(classes)
print(classes, available_classes)

last_class = 1
for c_index in range(len(available_classes)):
    clas = available_classes[c_index]
    for a_index in range(num_images):
        img_class = classes[a_index]
        if clas == img_class:
            rect = selected[a_index]
            pixelsInRect = img[rect.minY:rect.maxY, rect.minX:rect.maxX]
            show_image(pixelsInRect, "{} {}".format(a_index, last_class))
            break

"""
    if classes[index] > last_class:
        rect = selected[index]
        pixelsInRect = Irgb[rect.minY:rect.maxY, rect.minX:rect.maxX]
        show_image(pixelsInRect, "{} {}".format(index, last_class))

        last_class = classes[index]
"""

"""
print(len(final_selection))

for rect in final_selection:
    pixelsInRect = Irgb[rect.minY:rect.maxY, rect.minX:rect.maxX]
    show_image(pixelsInRect)
"""

cv2.destroyAllWindows()

# plt.figure(figsize=(15,10))
# plt.imshow(Irgb)
# plt.show()