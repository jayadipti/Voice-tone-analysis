# deepface_model.p

from deepface import DeepFace
import cv2

def analyze_emotion(frame):
    """
    Analyze the emotion of a given video frame using DeepFace.

    Parameters:
    frame (np.array): An image frame from the video or camera.

    Returns:
    str: The dominant emotion detected.
    """
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return "Unknown"

def start_camera_emotion_recognition():
    """
    Start the webcam feed and perform real-time facial emotion recognition.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam emotion recognition. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        emotion = analyze_emotion(frame)
        # Display the detected emotion on the frame
        cv2.putText(frame, f'Emotion: {emotion}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Facial Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_emotion_recognition()
