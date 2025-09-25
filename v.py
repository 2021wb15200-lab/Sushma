import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Get screen size
screen_width, screen_height = pyautogui.size()

# Capture webcam
cap = cv2.VideoCapture(0)

# Smooth pointer movement
prev_x, prev_y = 0, 0
smoothening = 7

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip and convert to RGB
        image = cv2.flip(image, 1)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # Extract landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                if lm_list:
                    # Index finger tip = id 8, Middle finger tip = id 12
                    x1, y1 = lm_list[8]
                    x2, y2 = lm_list[12]

                    # Map index finger to screen coordinates
                    screen_x = np.interp(x1, (100, w - 100), (0, screen_width))
                    screen_y = np.interp(y1, (100, h - 100), (0, screen_height))
                    curr_x = prev_x + (screen_x - prev_x) / smoothening
                    curr_y = prev_y + (screen_y - prev_y) / smoothening
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                    # Detect fingers up
                    fingers = []
                    fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)  # Thumb
                    fingers += [1 if lm_list[t][1] < lm_list[t - 2][1] else 0 for t in [8, 12, 16, 20]]

                    # Left Click: Index + Middle close together
                    length = np.hypot(x2 - x1, y2 - y1)
                    if fingers[1] and fingers[2] and length < 30:
                        pyautogui.click()
                        time.sleep(0.3)

                    # Right Click: Only middle finger up
                    if fingers[2] == 1 and fingers[1] == 0:
                        pyautogui.rightClick()
                        time.sleep(0.3)

                    # Scroll: Thumb + Index up
                    if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:
                        pyautogui.scroll(20 if y1 < y2 else -20)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show camera feed
        cv2.imshow("Virtual Mouse", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
