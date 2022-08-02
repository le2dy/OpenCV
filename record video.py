import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened:
    path = "Images/video.avi"
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    size = (int(width), int(height))
    out = cv2.VideoWriter(path, fourcc, fps, size)

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("record", frame)
            out.write(frame)

            if cv2.waitKey(int(1000 / fps)) != -1:
                break
        else:
            print("NO frame")
            break
    out.release()
else:
    print("can't open camera!")
cap.release()
cv2.destroyAllWindows()
