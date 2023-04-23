import cv2


def debug_image(img, title=''):
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
