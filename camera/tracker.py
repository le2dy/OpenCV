import cv2
import numpy as np

trackers = [cv2.TrackerMIL_create,
            cv2.TrackerKCF_create,
            cv2.TrackerCSRT_create]
trackerIdx = 0
tracker = None
is_first = True

video_src = '../Images/img.gif'

cap = cv2.VideoCapture(video_src)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
name = 'api'

while cv2.waitKey(33) != 27:
    ret, frame = cap.read()
    if not ret:
        break

    draw = frame.copy()
    if tracker is None:
        cv2.putText(draw, "Press 'Space' to set ROI", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2,
                    cv2.LINE_AA)
    else:
        ok, box = tracker.update(frame)
        (x, y, w, h)= box

        if ok:
            cv2.rectangle(draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, 1)
        else:
            cv2.putText(draw, "Tracking Fail", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2,
                        cv2.LINE_AA)
    trackerName = tracker.__class__.__name__
    cv2.putText(draw, str(trackerIdx) + " : " + trackerName, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2,
                cv2.LINE_AA)

    cv2.imshow(name, draw)

    key = cv2.waitKey(delay)

    if key == ord(' ') or (video_src != 0 and is_first):
        is_first = False
        roi = cv2.selectROI(name, frame, False)
        if roi[2] and roi[3]:
            tracker = trackers[trackerIdx]()
            is_init = tracker.init(frame, roi)
    elif key in range(48, 51):
        trackerIdx = key - 48
        if box is not None:
            tracker = trackers[trackerIdx]()
            is_init = tracker.init(frame, box)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
