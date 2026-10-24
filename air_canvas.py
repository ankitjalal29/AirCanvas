
import cv2
import mediapipe as mp
import numpy as np


camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

canvas = None
previous_x, previous_y = 0, 0


colors = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255)     # Yellow
]
color_names = ["RED", "GREEN", "BLUE", "YELLOW"]
color_index = 0
brush_size = 6

while True:
    success, frame = camera.read()

    if not success:
        print("Camera not detected!")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            points = hand.landmark

     
            index_tip = points[8]
            middle_tip = points[12]

            index_x = int(index_tip.x * width)
            index_y = int(index_tip.y * height)
            middle_y = int(middle_tip.y * height)

            
            index_up = (
                points[8].y < points[6].y
            )
            middle_up = (
                points[12].y < points[10].y
            )

            if index_up and not middle_up:
                cv2.circle(
                    frame,
                    (index_x, index_y),
                    10,
                    colors[color_index],
                    -1
                )

                if previous_x != 0 and previous_y != 0:
                    cv2.line(
                        canvas,
                        (previous_x, previous_y),
                        (index_x, index_y),
                        colors[color_index],
                        brush_size
                    )

                previous_x, previous_y = index_x, index_y

            
            elif index_up and middle_up:
                previous_x, previous_y = 0, 0
                cv2.circle(
                    frame,
                    (index_x, index_y),
                    10,
                    (255, 255, 255),
                    2
                )
            else:
                previous_x, previous_y = 0, 0

    else:
        previous_x, previous_y = 0, 0

 
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    inverse_mask = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    drawing = cv2.bitwise_and(canvas, canvas, mask=mask)
    output = cv2.add(background, drawing)


    cv2.rectangle(output, (0, 0), (width, 65), (35, 35, 35), -1)
    cv2.putText(
        output, "AIR CANVAS", (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    for i, name in enumerate(color_names):
        x = 160 + i * 115
        cv2.circle(output, (x, 22), 9, colors[i], -1)
        cv2.putText(
            output, name, (x - 28, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1
        )

    cv2.putText(
        output,
        "C: Clear | 1-4: Colors | Q: Quit",
        (15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1
    )

    cv2.imshow("Air Canvas - Finger Drawing", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0
        previous_x, previous_y = 0, 0
    elif key in [ord("1"), ord("2"), ord("3"), ord("4")]:
        color_index = key - ord("1")

camera.release()
hands.close()
cv2.destroyAllWindows()
