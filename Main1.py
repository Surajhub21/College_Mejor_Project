import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os
import mediapipe as mp

# Function to load training data
def load_training_data(training_folder, image_size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(training_folder):
        label_folder = os.path.join(training_folder, label)
        if not os.path.isdir(label_folder):
            continue
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized_img = cv2.resize(img, image_size)
                images.append(resized_img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Function to preprocess images (flatten and normalize)
def preprocess_images(images):
    processed_images = []
    for img in images:
        flattened_img = img.flatten()
        processed_images.append(flattened_img)
    processed_images = np.array(processed_images)
    scaler = StandardScaler()
    processed_images = scaler.fit_transform(processed_images)
    return processed_images, scaler

# Set the correct path to the training data directory
training_folder = r'D:\All Programme File\Project\Mejor_Project\training_data'
train_images, train_labels = load_training_data(training_folder)
print(f"Number of training samples: {len(train_images)}")

# Preprocess the training images
X_train, scaler = preprocess_images(train_images)

# Ensure n_neighbors is not greater than the number of training samples
n_neighbors = min(3, len(train_images))

# Train K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, train_labels)

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Open the camera and perform real-time object recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box coordinates
            h, w, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(x_max, w))
            y_max = int(min(y_max, h))

            roi_gray = frame[y_min:y_max, x_min:x_max]
            
            if roi_gray.size == 0:
                print("Empty ROI extracted, skipping this hand.")
                continue
            
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

            # Resize and preprocess the ROI
            resized_roi = cv2.resize(roi_gray, (64, 64))
            processed_roi = resized_roi.flatten().reshape(1, -1)
            processed_roi = scaler.transform(processed_roi)

            # Predict the label of the object in the ROI
            prediction = knn.predict(processed_roi)
            predicted_label = prediction[0]

            # Calculate the probability of the prediction
            probabilities = knn.predict_proba(processed_roi)[0]
            max_prob = max(probabilities)
            confidence = f"{max_prob * 100:.2f}%"

            # Display the bounding box, label, and confidence
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, f"{predicted_label} ({confidence})", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Hand Sign Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
hands.close()


# Requirements
# requirements.txt
# absl-py==2.1.0
# astunparse==1.6.3
# certifi==2024.6.2
# charset-normalizer==3.3.2
# flatbuffers==24.3.25
# gast==0.5.4
# google-pasta==0.2.0
# grpcio==1.64.1
# h5py==3.11.0
# idna==3.7
# keras==3.3.3
# libclang==18.1.1
# Markdown==3.6
# markdown-it-py==3.0.0
# MarkupSafe==2.1.5
# mdurl==0.1.2
# ml-dtypes==0.3.2
# namex==0.0.8
# numpy==1.26.4
# opencv-python==4.10.0.84
# opt-einsum==3.3.0
# optree==0.11.0
# packaging==24.1
# protobuf==4.25.3
# Pygments==2.18.0
# requests==2.32.3
# rich==13.7.1
# six==1.16.0
# tensorboard==2.16.2
# tensorboard-data-server==0.7.2
# tensorflow==2.16.1
# tensorflow-intel==2.16.1
# tensorflow-io-gcs-filesystem==0.31.0
# termcolor==2.4.0
# typing_extensions==4.12.2
# urllib3==2.2.2
# Werkzeug==3.0.3
# wrapt==1.16.0
