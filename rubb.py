from cv2 import cv2


# show the coordinate when clicking the mouse
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return x, y
    else:
        return 0, 0


if __name__ == '__main__':
    img = cv2.imread('images/sample_out.jpg')
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', on_mouse)
    cv2.imshow('img', img)
    cv2.waitKey(0)
