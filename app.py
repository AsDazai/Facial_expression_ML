from flask import Flask, render_template, Response
import cv2
import numpy as np
from deepface import DeepFace
from collections import defaultdict
import atexit
import time

app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not access the camera!")
    exit()

emotion_history = []  # Store last 5 detected emotions
frame_count = 0  # Counter to run DeepFace every 10 frames

# Function to generate video frames
def generate_frames():
    global emotion_history, frame_count
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame!")
            break  # Stop loop if camera fails

        frame_count += 1

        try:
            if frame_count % 10 == 0:  # Run DeepFace every 10th frame
                result = DeepFace.analyze(frame, actions=['emotion'], 
                                          detector_backend="opencv",  # Faster face detection
                                          enforce_detection=False)

                if isinstance(result, list) and len(result) > 0:
                    dominant_emotion = result[0]['dominant_emotion']
                    confidence = result[0]['emotion'].get(dominant_emotion, 0)

                    # Only store emotions with confidence > 50%
                    if confidence > 50:
                        emotion_history.append((dominant_emotion, confidence))

                    if len(emotion_history) > 5:
                        emotion_history.pop(0)

            # Weighted emotion averaging
            emotion_weights = defaultdict(float)
            total_weight = sum(conf for _, conf in emotion_history)
            for emo, conf in emotion_history:
                emotion_weights[emo] += conf

            final_emotion = max(emotion_weights, key=emotion_weights.get) if total_weight > 0 else "Uncertain"

            # Display detected emotion
            cv2.putText(frame, f"Emotion: {final_emotion}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"DeepFace Error: {e}")

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit:
            print("Client disconnected, stopping stream.")
            break  # Stop sending frames if client disconnects

        time.sleep(0.05)  # Prevent excessive CPU usage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Release camera when app stops
def release_camera():
    if camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")

atexit.register(release_camera)

if __name__ == '__main__':
    app.run(debug=False, threaded=False, use_reloader=False)
