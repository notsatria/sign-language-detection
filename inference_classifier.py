import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Open webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # Convert the BGR image to RGB before processing.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data_aux = []

    H, W, _ = frame.shape

    x_ = []
    y_ = []

    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                  mp_drawing_styles.get_default_hand_connections_style())
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
            
        x1 = int(min(x_) * W) - 10
        x2 = int(max(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        y2 = int(max(y_) * H) - 10
        
        # Predict the gesture
        prediction = model.predict([np.asarray(data_aux)])

        # print('Prediction: {prediction}'.format(prediction = prediction[0]))

        prediction_character = labels_dict[int(prediction[0])]
    
        cv2.rectangle(frame, (x1, y1), (x2 , y2), (0, 0, 0), 2)

        cv2.putText(frame, prediction_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2, cv2.LINE_AA)

        print('Prediction: {prediction}'.format(prediction = prediction_character))

    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()