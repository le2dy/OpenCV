import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

hog_aim = cv2.HOGDescriptor((48, 96), (16, 16), (8, 8), (8, 8), 9)
hog_aim.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

cap = cv2.VideoCapture('../Images/img.gif')
mode = True

while cv2.waitKey(33) != 27:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1024, 640))

    if ret:
        if mode:
            found, _ = hog.detectMultiScale(frame)
            for (x, y, w, h) in found:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255))
        else:
            found, _ = hog_aim.detectMultiScale(frame)
            for (x, y, w, h) in found:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.putText(frame, "Detector:%s" % ('Default' if mode else 'Daimler'), (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                    1,(0, 255, 0), 1)
        cv2.imshow('frame', frame)

        if cv2.waitKey(33) == ord(' '):
            mode = not mode
    else:
        break

cap.release()
cv2.destroyAllWindows()
