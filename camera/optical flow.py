import cv2
import numpy as np

cap = cv2.VideoCapture('../Images/img.gif')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

color = np.random.randint(0, 255, (200, 3))
lines = None
prevImg = None
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, .03)

while cv2.waitKey(33) != 27:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()

    if not ret:
        break
    draw = frame.copy()
    gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)

    if prevImg is None:
        prevImg = gray
        lines = np.zeros_like(frame)
        prevPt = cv2.goodFeaturesToTrack(prevImg, 200, .01, 10)
    else:
        nextImg = gray
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPt, None, criteria=termcriteria)
        prevMv = prevPt[status == 1]
        nextMv = nextPt[status == 1]

        for i, (p, n) in enumerate(zip(prevMv, nextMv)):
            px, py = p.ravel()
            nx, ny = n.ravel()
            cv2.line(lines, (int(px), int(py)), (int(nx), int(ny)), color[i].tolist(), 2)
            cv2.circle(draw, (int(nx), int(ny)), 2, color[i].tolist(), -1)
        draw = cv2.add(draw, lines)
        prevImg = nextImg
        prevPt = nextPt.reshape(-1, 1, 2)
    cv2.imshow('optical flow', draw)
    if cv2.waitKey(33) == 8:
        prevImg = None

cap.release()
cv2.destroyAllWindows()
