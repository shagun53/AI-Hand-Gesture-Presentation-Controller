# Smart Presentation Controller using Hand Gesture Recognition

## Overview

This project uses Deep Learning and Computer Vision to recognize hand gestures and classify them into one of 20 gesture classes. The model is trained on 24,000 hand gesture images using TensorFlow and Convolutional Neural Networks (CNNs).

The long-term goal is to use recognized gestures to control presentation slides, enabling touch-free presentation navigation.

---

## Features

* Hand Gesture Recognition using CNN
* Trained on 24,000 images
* 20 Gesture Classes
* TensorFlow & Keras implementation
* Image-based gesture prediction
* Ready for real-time webcam integration
* Can be extended for presentation control using OpenCV and PyAutoGUI

---

## Dataset

Dataset: Hand Gesture Recognition Dataset

* 20 gesture classes
* 24,000 images
* Balanced dataset
* 1,200 images per class

Dataset Source:

https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset

---

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* OpenCV (future integration)
* MediaPipe (future integration)
* PyAutoGUI (future integration)

---

## Project Workflow

1. Load Dataset
2. Create CSV containing image paths and labels
3. Encode Labels
4. Split Dataset into Train and Test Sets
5. Preprocess Images
6. Build CNN Model
7. Train Model
8. Evaluate Performance
9. Save Trained Model
10. Test on New Images
11. Integrate with Webcam for Real-Time Recognition

---

## Model Architecture

```text
Input Image (224x224x3)
        │
Conv2D (32)
        │
MaxPooling2D
        │
Conv2D (64)
        │
MaxPooling2D
        │
Conv2D (128)
        │
MaxPooling2D
        │
Flatten
        │
Dense (128)
        │
Dropout (0.3)
        │
Dense (20 Softmax)
```

## Training Results

Training Accuracy: ~99.9%

Validation Accuracy: ~100%

Example Results:

| Image     | Predicted Class | Confidence |
| --------- | --------------- | ---------- |
| test1.jpg | 0               | 98.51%     |
| test2.jpg | 6               | 46.54%     |
| test3.jpg | 4               | 60.08%     |
| test4.jpg | 12              | 93.98%     |

---

## Project Structure

```text
Smart-Presentation-Controller/
│
├── Hand_Gesture_AI_Model.ipynb
├── Gesture_Prediction_Testing.ipynb
├── gesture_controller.keras
├── hand_gesture_dataset.csv
├── screenshots/
├── README.md
└── requirements.txt
```

## Load Trained Model

```python
from tensorflow.keras.models import load_model

model = load_model("gesture_controller.keras")
```

## Predict a Gesture

```python
prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(predicted_class)
print(confidence)
```

## Future Enhancements

* Real-time webcam gesture recognition
* MediaPipe hand tracking
* Gesture-to-action mapping
* Presentation slide navigation
* Virtual avatar control
* Gesture-based human-computer interaction

## Author

Developed as a Computer Vision and Deep Learning project using TensorFlow and CNNs for hand gesture recognition.
