import cv2


def debug_image(img, title=''):
    window_name = "image"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 20, 20)
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    k = None
    while k != 110:
        k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
