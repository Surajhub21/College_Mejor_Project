import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

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

# Function to preprocess images (flatten)
def preprocess_images(images):
    processed_images = []
    for img in images:
        flattened_img = img.flatten()
        processed_images.append(flattened_img)
    return np.array(processed_images)

# Set the correct path to the training data directory
training_folder = r'D:\All Programme File\Project\Mejor_Project\training_data'
train_images, train_labels = load_training_data(training_folder)
print(f"Number of training samples: {len(train_images)}")

# Preprocess the training images
X_train = preprocess_images(train_images)

# Ensure n_neighbors is not greater than the number of training samples
n_neighbors = min(3, len(train_images))

# Train K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, train_labels)

# Load a pre-trained Haar Cascade classifier for object detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade_path)

# Open the camera and perform real-time object recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects (e.g., faces) in the frame
    objects = detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

    for (x, y, w, h) in objects:
        # Extract the object region of interest (ROI)
        roi_gray = gray_frame[y:y+h, x:x+w]

        # Resize and preprocess the ROI
        resized_roi = cv2.resize(roi_gray, (64, 64))
        processed_roi = resized_roi.flatten()

        # Predict the label of the object in the ROI
        prediction = knn.predict([processed_roi])
        predicted_label = prediction[0]

        # Display the bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Object Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
