import os

import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)


DATA_DIR = './data'

data = []
labels = []

for dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
        # Convert the BGR image to RGB before processing.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        result = hands.process(img_rgb)
        if (result.multi_hand_landmarks):
            for hand_landmarks in result.multi_hand_landmarks:
                # See how the landmarks look like
                """
                mp_drawing.draw_landmarks(
                    img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                      mp_drawing_styles.get_default_hand_connections_style())
                """
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            
            data.append(data_aux)
            labels.append(dir)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()