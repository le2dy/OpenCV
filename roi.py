import cv2

ox, oy, px, py = 0, 0, 0, 0
isClick = False

src = cv2.imread("Images/road.png")


def onMouse(event, x, y, flag, param):
    global ox, oy, src, px, py, isClick

    if event == cv2.EVENT_LBUTTONDOWN:
        ox, oy = x, y
        isClick = True
    elif event == cv2.EVENT_MOUSEMOVE and isClick:
        px, py = x, y
        w = ox - px
        h = oy - py
        cp = src.copy()
        cv2.rectangle(cp, (ox, oy), (ox - w, oy - h), (0, 0, 255), 2)
        cv2.imshow('src', cp)
    elif event == cv2.EVENT_LBUTTONUP:
        isClick = False
        w = ox - px
        h = oy - py
        cv2.rectangle(src, (ox, oy), (ox - w, oy - h), (255, 0, 0), 2)
        cv2.imshow('src', src)


cv2.imshow('src', src)
cv2.setMouseCallback('src', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()
