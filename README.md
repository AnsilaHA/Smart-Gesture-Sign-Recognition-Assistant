# Smart Gesture & Sign Recognition Assistant

## Problem Statement

Many people with hearing and speech disabilities use sign language for communication. However, most people do not understand sign language, creating communication barriers. 

Our project solves this problem by automatically recognizing hand gestures and converting them into understandable text and voice output.

## Main Idea & Architecture

- **Hybrid Architecture:** The system uses a MediaPipe-driven hybrid architecture.
- **Data Collection:** The webcam captures hand gestures. MediaPipe detects the hand landmarks in real-time.
- **Skeleton-on-Black Pipeline:** To ensure invariance to lighting, background noise, and skin tone, we extract a "Skeleton-on-Black" image classification pipeline. Hand landmarks are drawn on a black background and cropped.
- **CNN Model:** A Convolutional Neural Network (CNN) predicts the gesture from the Skeleton-on-Black images.
- **Feedback:** The recognized gesture is displayed as text and converted into speech using `pyttsx3`.

## Technologies Used

- **Python** for programming
- **OpenCV** for webcam handling and image processing
- **MediaPipe** for hand tracking and landmark extraction
- **TensorFlow/Keras** for CNN-based gesture recognition
- **Scikit-Learn** for robust dataset splitting
- **pyttsx3** & **pywin32** for text-to-speech conversion
- **Streamlit** for the user interface

## Setup Instructions

1. **Install Python**
   Ensure you have Python 3.8 to 3.10 installed on your system (TensorFlow is best supported on these versions).

2. **Open Command Prompt / Terminal**
   Navigate to the project folder:
   ```cmd
   cd path\to\smart-gesture-recognition-master
   ```

3. **Install Dependencies**
   Run the following command to install the required libraries:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run the Application**
   Start the Streamlit web application by running:
   ```cmd
   streamlit run app.py
   ```

## Usage (3-Step Pipeline)
The Streamlit interface offers a complete end-to-end pipeline:

### 1. Collect Data
- Enter a gesture name (e.g., "Hello", "Yes", "No").
- Click "Start Recording" to capture 500 frames of your gesture.
- Ensure your hand is visible; the system will automatically extract your hand landmarks and save "Skeleton-on-Black" images.

### 2. Train Model
- Once you have collected at least 2 gestures, go to the "Train Model" tab.
- Click "Train Model" to compile and train the CNN on your custom dataset.
- The system will train robustly using proper shuffling, optimized data augmentation, and early stopping, saving the model in the stable `.keras` format (`gesture_model.keras`).

### 3. Real-Time Recognition
- Go to the "Real-Time Recognition" tab.
- Check the **"Start Webcam"** box.
- Show your hand to the webcam. The system will predict the gesture, display the text, and read it out loud.

## Advantages

- Works in real time
- Invariant to lighting, background noise, and skin tone via the Skeleton-on-Black approach
- End-to-end custom dataset collection and training UI
- Helps deaf and mute people improve accessibility
- Uses AI and Deep Learning
- Low-cost and user-friendly

## Conclusion

In conclusion, our project demonstrates how Deep Learning and Computer Vision can be used to build an intelligent gesture recognition system that improves communication and accessibility.
