import cv2

cap = cv2.VideoCapture('../Images/img.gif')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

fgbg = cv2.createBackgroundSubtractorMOG2()

while cv2.waitKey(33) != 27:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()

    fgmask = fgbg.apply(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('fgbg', fgmask)

cap.release()
cv2.destroyAllWindows()
