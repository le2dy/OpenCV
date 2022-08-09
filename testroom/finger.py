import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
finger_coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_coord = (4, 2)

while True:
    ret, frame = cap.read()
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(RGB)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handList = []
        for handLms in multiLandMarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handList.append((cx, cy))

        for point in handList:
            cv2.circle(frame, point, 10, (225, 255, 0), cv2.FILLED)

        up_count = 0

        for coordinate in finger_coord:
            if handList[coordinate[0]][1] < handList[coordinate[1]][1]:
                up_count += 1
        if handList[thumb_coord[0]][0] > handList[thumb_coord[1]][0]:
            up_count += 1

        cv2.putText(frame, str(up_count), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, ))
    cv2.imshow('Finger', frame)
    if cv2.waitKey(33) == 27:
        break
cv2.destroyAllWindows()
