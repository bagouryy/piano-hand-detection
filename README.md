# Real-time Hand Gesture Piano

## Description

The "Real-time Hand Gesture Piano" project is a Python-based application that leverages computer vision and machine learning to enable users to play a virtual piano using hand gestures. The application utilizes the MediaPipe library for hand tracking and OpenCV for image processing. The piano interface is overlaid on the webcam feed, allowing users to play different notes by interacting with the virtual keys using their hands.

## Features

1. **Real-time Hand Tracking:** The application uses the MediaPipe library to accurately track the user's hand movements in real-time.

2. **Virtual Piano Overlay:** A virtual piano is overlaid on the webcam feed, providing a visual representation of the piano keys.

3. **Gesture-based Interaction:** Users can play different piano notes by performing specific hand gestures corresponding to different keys on the virtual piano.

4. **Multithreading:** The application utilizes multithreading to ensure smooth audio playback while continuously tracking hand gestures.

5. **Audio Feedback:** The playsound library is employed to provide audio feedback, playing the corresponding piano notes when a user triggers the associated hand gesture.

6. **Dynamic Key Mapping:** The hand gestures are mapped to specific piano notes, allowing users to create music dynamically by moving their hands within the defined regions.

## How to Use

1. Run the script, ensuring that the required libraries (OpenCV, MediaPipe, playsound) are installed.
2. The webcam feed will display in real-time with the virtual piano overlay.
3. Perform specific hand gestures over the virtual keys to play different piano notes.
4. Experiment with different hand movements to create unique musical compositions.

## Dependencies

- OpenCV
- MediaPipe
- playsound

**Note:** Ensure that the necessary audio files for each piano note are available in the specified folder ("audio/piano3/") for the application to function correctly.

## Author

Nour Albagoury

## GitHub Repository

https://github.com/bagouryy/PianoHandDetection
