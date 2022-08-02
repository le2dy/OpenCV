import cv2

cascade_file = '../haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(gray, minSize=(5,5))

    for (x, y, w, h) in face_list:
        roi = frame[y:y + h, x:x + w]
        frame[y:y + h, x:x + w] = cv2.blur(roi, (100, 100))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness = 3)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(33)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
