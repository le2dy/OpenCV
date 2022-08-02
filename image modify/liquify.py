import cv2
import numpy as np

name = 'liquify'
half = 50
isDrag = False


def liquify(src, cx1, cy1, cx2, cy2):
    x, y, w, h = cx1 - half, cy1 - half, half * 2, half * 2
    roi = src[y:y + h, x:x + w].copy()
    out = roi.copy()

    offset_cx1, offset_cy1 = cx1 - x, cy1 - y
    offset_cx2, offset_cy2 = cx2 - x, cy2 - y

    # 변환 전
    tri1 = [[[0, 0], [w, 0], [offset_cx1, offset_cy1]],
            [[0, 0], [0, h], [offset_cx1, offset_cy1]],
            [[w, 0], [offset_cx1, offset_cy1], [w, h]],
            [[0, h], [offset_cx1, offset_cy1], [w, h]]]

    # 변환 후
    tri2 = [[[0, 0], [w, 0], [offset_cx2, offset_cy2]],
            [[0, 0], [0, h], [offset_cx2, offset_cy2]],
            [[w, 0], [offset_cx2, offset_cy2], [w, h]],
            [[0, h], [offset_cx2, offset_cy2], [w, h]]]

    for i in range(4):
        matrix = cv2.getAffineTransform(np.float32(tri1[i]), np.float32(tri2[i]))
        warped = cv2.warpAffine(roi.copy(), matrix, (w, h), None, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((w, h), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri2[i]), (255, 255, 255))

        warped = cv2.bitwise_and(warped, warped, mask=mask)
        out = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(mask))
        out += warped

    src[y:y + h, x:x + w] = out
    return src


def onMouse(event, x, y, flag, params):
    global cx1, cy1, isDrag, src

    if event == cv2.EVENT_LBUTTONDOWN and event == cv2.EVENT_MOUSEMOVE:
        print('a')

    if event == cv2.EVENT_MOUSEMOVE:
        if not isDrag:
            src_draw = src.copy()
            cv2.rectangle(src_draw, (x - half, y - half), (x + half, y + half), (0, 255, 0))
            cv2.imshow(name, src_draw)
    elif event == cv2.EVENT_LBUTTONDOWN:
        isDrag = True
        cx1, cy1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        if isDrag:
            isDrag = False
            liquify(src, cx1, cy1, x, y)
            cv2.imshow(name, src)

if __name__ == '__main__':
    src = cv2.imread('../Images/apple.png')
    h, w = src.shape[:2]
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, onMouse)
    cv2.imshow(name, src)
    while True:
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
cv2.destroyAllWindows()
