import torch
import cv2
from fer import FER

# Load FER model
emotion_detector = FER()#fer library is instantiated, loading a pre-trained emotion detection model

# Capture video
cap = cv2.VideoCapture(0)#initializes the webcam feed (usually 0 is for the default webcam)

while True:
    ret, frame = cap.read()#loop continuously captures video frames until manually stopped.
    if not ret:#ret: A Boolean indicating whether the frame was successfully captured.frame: The actual frame image
        break

    # Perform emotion detection
    result = emotion_detector.detect_emotions(frame)#detect_emotions(frame): The FER model analyzes the current video frame, trying to detect faces and emotions. It returns a list of detected faces with associated emotion scores.

    if result: #IF FACE ALREADY DETECTED
        # Draw bounding boxes and label emotion
        for face in result:
            (x, y, w, h) = face['box']#bounding box (coordinates x, y, width w, and height h) of the detected face.
            emotion = face['emotions']# Extracts the emotions and their respective confidence scores.
            dominant_emotion = max(emotion, key=emotion.get) #finds the emotion with the highest confidence score for the current face, identifying the dominant emotion.

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draw Rectangle Around Faces and Display Emotion
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('FER+ Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()