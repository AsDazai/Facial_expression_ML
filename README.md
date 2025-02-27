# Real-Time Facial Emotion Detection

## Created by
AsDazai

## Description
This project is a real-time facial emotion detection system using DeepFace and OpenCV. It captures live video from the webcam, processes frames to detect facial expressions, and displays the detected emotions on the screen. The model is optimized to improve accuracy by using weighted emotion averaging and an adaptive confidence threshold. This system can be used for various applications such as sentiment analysis, human-computer interaction, and emotion-based user experiences.

## Technologies Used
- Python
- Flask
- OpenCV
- DeepFace
- NumPy

## Working
1. The Flask server runs a web application that streams live video.
2. OpenCV captures frames from the webcam.
3. DeepFace analyzes emotions every 10th frame using the RetinaFace detector.
4. Detected emotions are stored, and a weighted averaging technique determines the most accurate emotion.
5. The emotion is displayed on the video stream in real-time.

